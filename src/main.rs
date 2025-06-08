mod image_processor;
#[cfg(test)]
mod tests;

use anyhow::{Result, Context};
use clap::Parser;
use image::DynamicImage;
use ndarray::{Array, Array2, Array4};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use reqwest::blocking::Client;
use std::path::PathBuf;
use std::fs;
use tokenizers::Tokenizer;
use std::collections::HashMap;

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

    /// URL of the image to process
    #[arg(long, default_value = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")]
    image_url: String,

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
}

impl SmolVLM {
    fn new(
        vision_model_path: &PathBuf,
        embed_model_path: &PathBuf,
        decoder_model_path: &PathBuf,
        tokenizer_path: &PathBuf,
    ) -> Result<Self> {
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        // Get model configuration from the decoder session
        let config = SmolVLMConfig {
            num_key_value_heads: 5,  // Match the model's expected number of heads
            head_dim: 120,          // This needs to match the model's configuration
            num_hidden_layers: 13,   // This needs to match the model's configuration
            eos_token_id: 2,
            image_token_id: 33280,
        };

        // Debug: Print tokenizer file contents
        println!("Reading tokenizer file from: {}", tokenizer_path.display());
        let tokenizer_contents = fs::read_to_string(tokenizer_path)
            .context("Failed to read tokenizer file")?;
        println!("Tokenizer file contents (first 1000 chars):\n{}", 
            tokenizer_contents.chars().take(1000).collect::<String>());

        let processor = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}\nFile contents preview: {}", 
                e, 
                tokenizer_contents.chars().take(1000).collect::<String>()))?;
        let image_processor = SmolVLMImageProcessor::new();

        // Use configuration that matches the model's expectations
        let config = SmolVLMConfig {
            num_key_value_heads: 5,
            head_dim: 64,
            num_hidden_layers: 32,
            eos_token_id: 2,
            image_token_id: 49190,
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

    fn load_image(url: &str) -> Result<DynamicImage> {
        let client = Client::new();
        let response = client.get(url).send()?;
        let image_data = response.bytes()?;
        let image = image::load_from_memory(&image_data)?;
        Ok(image)
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
        let mut current_attention_mask = attention_mask.iter().map(|&x| x as u32).collect::<Vec<_>>();
        let mut current_input_ids = input_ids.iter().map(|&x| x as u32).collect::<Vec<_>>();
        
        // 4. Prepare decoder inputs
        let batch_size = 1;
        let mut past_key_values = HashMap::new();
        for layer in 0..self.config.num_hidden_layers {
            // Initialize key and value arrays with correct dimensions
            let key_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim));
            let value_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim));
            
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
        let (processed_image, attention_mask) = processed_image;
        vision_inputs.insert("pixel_values", Value::from_array(processed_image)?.into());
        vision_inputs.insert("pixel_attention_mask", Value::from_array(attention_mask)?.into());
        
        let vision_outputs = self.vision_session.run(vision_inputs)?;
        let image_features = vision_outputs[0].try_extract_tensor::<f32>()?.to_owned();

        // 6. Generation loop
        let max_new_tokens = 1024;
        let mut generated_tokens = Vec::new();
        let mut position_ids: Vec<u32> = (0..current_input_ids.len() as u32).collect();

        for _ in 0..max_new_tokens {
            // Get input embeddings
            let mut embed_inputs = HashMap::new();
            let input_ids_i64: Vec<i64> = current_input_ids.iter().map(|&x| x as i64).collect();
            let len = input_ids_i64.len();
            let input_ids_array = Array::from_vec(input_ids_i64)
                .into_shape((1, len))?
                .into_owned();
            embed_inputs.insert("input_ids", Value::from_array(input_ids_array.clone())?);
            
            println!("\nEmbedding model input:");
            println!("input_ids shape: {:?}", input_ids_array.shape());
            println!("input_ids type: i64");
            
            let embed_outputs = self.embed_session.run(embed_inputs)?;
            let mut input_embeds = embed_outputs[0].try_extract_tensor::<f32>()?.to_owned();
            
            println!("Embedding model output:");
            println!("embeddings shape: {:?}", input_embeds.shape());
            println!("embeddings type: f32");

            // Replace image token embeddings with image features
            for (i, &token_id) in current_input_ids.iter().enumerate() {
                if token_id == self.config.image_token_id {
                    let mut slice = input_embeds.slice_mut(ndarray::s![i, ..]);
                    slice.assign(&image_features.slice(ndarray::s![0, ..]));
                }
            }

            // Prepare decoder inputs
            let mut decoder_inputs: HashMap<&str, Value> = HashMap::new();
            
            // inputs_embeds is f32
            decoder_inputs.insert("inputs_embeds", Value::from_array(input_embeds.clone())?.into());
            
            println!("\nDecoder model inputs:");
            println!("inputs_embeds shape: {:?}", input_embeds.shape());
            println!("inputs_embeds type: f32");
            
            // attention_mask is i64
            let attention_mask_i64: Vec<i64> = current_attention_mask.iter().map(|&x| x as i64).collect();
            let attention_mask_array = Array::from_vec(attention_mask_i64)
                .to_shape((1, current_attention_mask.len()))?.into_owned();
            decoder_inputs.insert("attention_mask", Value::from_array(attention_mask_array.clone())?.into());
            
            println!("attention_mask shape: {:?}", attention_mask_array.shape());
            println!("attention_mask type: i64");
            
            // position_ids is i64
            let position_ids_i64: Vec<i64> = position_ids.iter().map(|&x| x as i64).collect();
            let position_ids_array = Array::from_vec(position_ids_i64)
                .to_shape((1, position_ids.len()))?.into_owned();
            decoder_inputs.insert("position_ids", Value::from_array(position_ids_array.clone())?.into());
            
            println!("position_ids shape: {:?}", position_ids_array.shape());
            println!("position_ids type: i64");
            
            // Add past key values
            for (key, value) in &past_key_values {
                decoder_inputs.insert(key, Value::from_array(value.clone())?.into());
            }

            // Run decoder
            let decoder_outputs = self.decoder_session.run(decoder_inputs)?;

            // Get logits and update past key values
            let logits = decoder_outputs[0].try_extract_tensor::<f32>()?.to_owned();
            let present_key_values = decoder_outputs[1].try_extract_tensor::<f32>()?.to_owned();

            println!("\nPresent key values shape: {:?}", present_key_values.shape());

            // Get next token
            let logits_slice = logits.slice(ndarray::s![-1, ..]);
            let next_token = logits_slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow::anyhow!("Failed to find max logit"))?;
            
            generated_tokens.push(next_token);

            // Check for EOS token
            if next_token == self.config.eos_token_id {
                break;
            }

            // Update inputs for next iteration
            current_input_ids = vec![next_token];
            current_attention_mask = vec![1];
            position_ids = vec![position_ids.last().unwrap() + 1];

            // Update past key values
            for i in 0..self.config.num_hidden_layers {
                for kv in &["key", "value"] {
                    let key = format!("past_key_values.{}.{}", i, kv);
                    if let Some(value) = past_key_values.get_mut(&key) {
                        // Get the appropriate slice based on whether it's a key or value
                        let present_slice = if *kv == "key" {
                            present_key_values.slice(ndarray::s![0, i, .., ..])
                        } else {
                            present_key_values.slice(ndarray::s![0, i, .., ..])
                        };
                        value.assign(&present_slice);
                    }
                }
            }
        }

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

    let image = SmolVLM::load_image(&args.image_url)?;
    let response = model.generate(&args.prompt, image)?;
    println!("{}", response);

    Ok(())
}
