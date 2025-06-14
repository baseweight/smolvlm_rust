mod image_processor;
#[cfg(test)]
mod tests;

use anyhow::{Result, Context};
use clap::Parser;
use image::DynamicImage;
use ndarray::Array;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
};
use reqwest::blocking::Client;
use std::path::PathBuf;
use std::fs;
use std::io::Write;
use tokenizers::Tokenizer;
use std::collections::HashMap;
use rand;

use crate::image_processor::SmolVLMImageProcessor;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the vision encoder ONNX model
    #[arg(long, default_value = "onnx_models/vision_encoder.onnx")]
    vision_model: PathBuf,

    /// Path to the token embedding ONNX model
    #[arg(long, default_value = "onnx_models/embed_tokens.onnx")]
    embed_model: PathBuf,

    /// Path to the decoder ONNX model
    #[arg(long, default_value = "onnx_models/decoder_model_merged.onnx")]
    decoder_model: PathBuf,

    /// Path to the tokenizer file
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer: PathBuf,

    /// URL of the image to process (optional)
    #[arg(long)]
    image_url: Option<String>,

    /// Path to a local image file (optional)
    #[arg(long, default_value = "boat.png")]
    image_path: PathBuf,

    /// Prompt to use for generation
    #[arg(long, default_value = "Can you describe this image?")]
    prompt: String,
}

struct SmolVLM {
    vision_session: Session,
    embed_session: Session,
    decoder_session: Session,
    processor: Tokenizer,
    image_processor: SmolVLMImageProcessor,
    config: SmolVLMConfig,
}

struct SmolVLMConfig {
    num_key_value_heads: usize,
    head_dim: usize,
    num_hidden_layers: usize,
    eos_token_id: u32,
    image_token_id: u32,
    max_context_length: usize,
}

impl SmolVLM {
    fn new(
        vision_model_path: &PathBuf,
        embed_model_path: &PathBuf,
        decoder_model_path: &PathBuf,
        tokenizer_path: &PathBuf,
    ) -> Result<Self> {
        // Check CUDA availability first
        let cuda = CUDAExecutionProvider::default();
        match cuda.is_available() {
            Ok(true) => println!("CUDA is available"),
            Ok(false) => {
                println!("CUDA is not available - please ensure CUDA 12 and cuDNN 9.x are installed");
                std::process::exit(1);
            },
            Err(e) => {
                println!("Error checking CUDA availability: {}", e);
                std::process::exit(1);
            }
        }

        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        let processor = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let image_processor = SmolVLMImageProcessor::new();

        // Use configuration that matches the model's expectations
        let config = SmolVLMConfig {
            num_key_value_heads: 5,
            head_dim: 64,
            num_hidden_layers: 32,
            eos_token_id: 2,
            image_token_id: 49190,
            max_context_length: 2048,
        };

        Ok(Self {
            vision_session,
            embed_session,
            decoder_session,
            processor,
            image_processor,
            config,
        })
    }

    fn load_image(url_or_path: &str) -> Result<DynamicImage> {
        if url_or_path.starts_with("http://") || url_or_path.starts_with("https://") {
            let client = Client::new();
            let response = client.get(url_or_path).send()?;
            let image_data = response.bytes()?;
            let image = image::load_from_memory(&image_data)?;
            Ok(image)
        } else {
            let image = image::open(url_or_path)?;
            Ok(image)
        }
    }

    fn generate(&self, prompt: &str, image: DynamicImage) -> Result<String> {
        // 1. Process the image
        let processed_image = self.image_processor.preprocess(image)?;
        
        // 2. Create the chat template
        let messages = format!(
            r#"<|im_start|>user
<|im_start|>image<|im_end|>
<|im_start|>text
{prompt}<|im_end|>
<|im_start|>assistant
"#
        );
        
        // 3. Tokenize the input
        let encoding = self.processor.encode(messages, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        
        // Convert attention mask to Vec
        let attention_mask = attention_mask.iter().map(|&x| x as u32).collect::<Vec<_>>();
        let input_ids = input_ids.iter().map(|&x| x as u32).collect::<Vec<_>>();
        
        // 4. Prepare decoder inputs
        let batch_size = 1;
        let mut past_key_values: HashMap<String, Array<f32, _>> = HashMap::new();
        for layer in 0..self.config.num_hidden_layers {
            // Initialize key and value arrays with dynamic shapes
            let key_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            let value_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            
            // Insert both key and value arrays
            past_key_values.insert(
                format!("past_key_values.{}.key", layer),
                key_array,
            );
            past_key_values.insert(
                format!("past_key_values.{}.value", layer),
                value_array,
            );
        }

        // 5. Get image features
        let mut vision_inputs: HashMap<&str, Value> = HashMap::new();
        
        // Convert the processed image tuple into separate values
        let (processed_image, pixel_attention_mask) = processed_image;
        vision_inputs.insert("pixel_values", Value::from_array(processed_image.clone())?.into());
        
        // Convert attention mask to boolean while maintaining shape (batch, num_frames, height, width)
        let pixel_attention_mask_bool = pixel_attention_mask.map(|&x| x != 0);
        vision_inputs.insert("pixel_attention_mask", Value::from_array(pixel_attention_mask_bool.clone())?.into());
        
        let vision_outputs = self.vision_session.run(vision_inputs)?;
        let image_features = vision_outputs[0].try_extract_tensor::<f32>()?.to_owned();

        // 6. Generation loop
        let max_new_tokens = 1024;
        let mut generated_tokens = Vec::new();
        let mut input_ids = Array::from_vec(input_ids.iter().map(|&x| x as i64).collect())
            .into_shape_with_order((1, input_ids.len()))?
            .into_owned();
        let mut attention_mask = Array::from_vec(attention_mask.iter().map(|&x| x as i64).collect())
            .into_shape_with_order((1, attention_mask.len()))?
            .into_owned();
        let mut position_ids = Array::from_vec((0..input_ids.len()).map(|x| x as i64).collect())
            .into_shape_with_order((1, input_ids.len()))?
            .into_owned();
        let mut image_features = None;

        for _ in 0..max_new_tokens {
            // Get input embeddings
            let mut embed_inputs: HashMap<&str, Value> = HashMap::new();
            embed_inputs.insert("input_ids", Value::from_array(input_ids.clone())?.into());
            let embed_outputs = self.embed_session.run(embed_inputs)?;
            let mut input_embeds = embed_outputs[0].try_extract_tensor::<f32>()?.to_owned();

            // Only compute vision features if not already computed
            if image_features.is_none() {
                let mut vision_inputs: HashMap<&str, Value> = HashMap::new();
                vision_inputs.insert("pixel_values", Value::from_array(processed_image.clone())?.into());
                vision_inputs.insert("pixel_attention_mask", Value::from_array(pixel_attention_mask_bool.clone())?.into());
                
                let vision_outputs = self.vision_session.run(vision_inputs)?;
                image_features = Some(vision_outputs[0].try_extract_tensor::<f32>()?.to_owned());
            }

            // Replace image token embeddings with image features
            for i in 0..input_ids.shape()[1] {
                if input_ids[[0, i]] == self.config.image_token_id as i64 {
                    let mut slice = input_embeds.slice_mut(ndarray::s![0, i, ..]);
                    slice.assign(&image_features.as_ref().unwrap().slice(ndarray::s![0, ..]));
                }
            }

            // Prepare decoder inputs
            let mut decoder_inputs: HashMap<&str, Value> = HashMap::new();
            decoder_inputs.insert("inputs_embeds", Value::from_array(input_embeds.clone())?.into());
            decoder_inputs.insert("attention_mask", Value::from_array(attention_mask.clone())?.into());
            decoder_inputs.insert("position_ids", Value::from_array(position_ids.clone())?.into());
            
            // Add past key values
            for (key, value) in &past_key_values {
                decoder_inputs.insert(key, Value::from_array(value.clone())?.into());
            }

            // Run decoder
            let decoder_outputs = self.decoder_session.run(decoder_inputs)?;
            let logits = decoder_outputs[0].try_extract_tensor::<f32>()?.to_owned();
            
            // Get next token from last position - exactly like Python version
            let last_idx = logits.shape()[1] - 1;
            let logits_slice = logits.slice(ndarray::s![0, last_idx, ..]);
            let next_token = logits_slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .ok_or_else(|| anyhow::anyhow!("Failed to find max logit"))?;
            
            // Update inputs for next iteration
            input_ids = Array::from_vec(vec![next_token]).into_shape_with_order((1, 1))?;
            attention_mask = Array::ones((1, 1));
            position_ids = position_ids.slice(ndarray::s![.., -1..]).map(|&x| x + 1);

            // Update past key values
            for i in 0..self.config.num_hidden_layers {
                let key = format!("past_key_values.{}.key", i);
                let value = format!("past_key_values.{}.value", i);
                
                if let Some(past_key) = past_key_values.get_mut(&key) {
                    let present_key = decoder_outputs[i * 2 + 1].try_extract_tensor::<f32>()?.to_owned();
                    let present_key = present_key.into_dyn();
                    *past_key = present_key;
                }
                
                if let Some(past_value) = past_key_values.get_mut(&value) {
                    let present_value = decoder_outputs[i * 2 + 2].try_extract_tensor::<f32>()?.to_owned();
                    let present_value = present_value.into_dyn();
                    *past_value = present_value;
                }
            }

            // Add to generated tokens
            generated_tokens.push(next_token as u32);

            // Check for EOS token
            if next_token == self.config.eos_token_id as i64 {
                break;
            }

            // Print token as we generate (streaming)
            if let Ok(token_str) = self.processor.decode(&[next_token as u32], true) {
                print!("{}", token_str);
            }
        }
        println!();

        // 7. Decode the generated tokens
        let generated_text = self.processor.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;
        Ok(generated_text)
    }
}

fn download_tokenizer(path: &PathBuf) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    println!("Downloading tokenizer...");
    let client = Client::new();
    let response = client
        .get("https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/tokenizer.json")
        .send()?;
    
    let tokenizer_data = response.bytes()?;
    fs::write(path, tokenizer_data)?;
    println!("Tokenizer downloaded successfully");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Download tokenizer if it doesn't exist
    download_tokenizer(&args.tokenizer)?;

    let model = SmolVLM::new(
        &args.vision_model,
        &args.embed_model,
        &args.decoder_model,
        &args.tokenizer,
    )?;

    // Use either the URL or local path for the image
    let image = if let Some(url) = args.image_url {
        SmolVLM::load_image(&url)?
    } else {
        SmolVLM::load_image(args.image_path.to_str().unwrap())?
    };
    
    let response = model.generate(&args.prompt, image)?;
    println!("{}", response);

    Ok(())
}
