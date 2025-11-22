mod image_processor;

use anyhow::{Result, Context};
use clap::Parser;
use image::DynamicImage;
use ndarray::{Array, Array4, Array5};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
    execution_providers::ExecutionProvider,
};

#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(target_os = "linux")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;
use reqwest::blocking::Client;
use std::path::PathBuf;
use std::fs;
use std::io::{self, Write};
use tokenizers::Tokenizer;
use std::collections::HashMap;

use crate::image_processor::SmolVLMImageProcessor;

#[derive(Parser, Debug)]
#[command(author = "SmolVLM", version = "0.1.0", about = "SmolVLM inference", long_about = None)]
struct Args {
    /// Path to the vision encoder ONNX model
    #[arg(long, default_value = "onnx_models/vision_encoder_q4.onnx")]
    vision_model: PathBuf,

    /// Path to the token embedding ONNX model
    #[arg(long, default_value = "onnx_models/embed_tokens_q4.onnx")]
    embed_model: PathBuf,

    /// Path to the decoder ONNX model
    #[arg(long, default_value = "onnx_models/decoder_model_merged_q4.onnx")]
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
    #[arg(long, default_value = "<|im_start|>User:<image>Can you describe this image?<end_of_utterance>\nAssistant:")]
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
        // Check GPU availability based on platform
        #[cfg(target_os = "windows")]
        {
            let provider = DirectMLExecutionProvider::default();
            match provider.is_available() {
                Ok(true) => println!("DirectML is available"),
                Ok(false) => {
                    println!("DirectML is not available - please ensure you have DirectX 12 support");
                    std::process::exit(1);
                },
                Err(e) => {
                    println!("Error checking DirectML availability: {}", e);
                    std::process::exit(1);
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            let provider = CUDAExecutionProvider::default();
            match provider.is_available() {
                Ok(true) => println!("CUDA is available"),
                Ok(false) => {
                    println!("CUDA is not available - please ensure CUDA 12+ and cuDNN are installed");
                    std::process::exit(1);
                },
                Err(e) => {
                    println!("Error checking CUDA availability: {}", e);
                    std::process::exit(1);
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            let provider = CoreMLExecutionProvider::default();
            match provider.is_available() {
                Ok(true) => println!("CoreML is available"),
                Ok(false) => {
                    println!("CoreML is not available");
                    std::process::exit(1);
                },
                Err(e) => {
                    println!("Error checking CoreML availability: {}", e);
                    std::process::exit(1);
                }
            }
        }

        // Create sessions with platform-specific execution providers
        #[cfg(target_os = "windows")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "linux")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "macos")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CoreMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "windows")]
        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        #[cfg(target_os = "linux")]
        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        #[cfg(target_os = "macos")]
        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CoreMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        #[cfg(target_os = "windows")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        #[cfg(target_os = "linux")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        #[cfg(target_os = "macos")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CoreMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        let processor = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let image_processor = SmolVLMImageProcessor::new();

        // Get special tokens from tokenizer
        let image_token = "<image>";
        let eos_token = "<end_of_utterance>";  // This is for SmolVLM2
        let image_token_id = processor.token_to_id(image_token)
            .ok_or_else(|| anyhow::anyhow!("Failed to get image token ID"))?;
        let _eos_token_id = processor.token_to_id(eos_token)
            .ok_or_else(|| anyhow::anyhow!("Failed to get EOS token ID"))?;

        // Override with correct EOS token ID from Python example
        let eos_token_id = 2;

        println!("Image token ID: {}", image_token_id);
        println!("EOS token ID: {}", eos_token_id);

        // Auto-detect model configuration from decoder inputs
        let decoder_inputs = &decoder_session.inputs;
        let mut max_layer = 0;
        let detected_kv_heads = 5; // Based on decoder model requirements

        for input in decoder_inputs {
            if input.name.starts_with("past_key_values.") {
                if let Some(layer_str) = input.name.split('.').nth(1) {
                    if let Ok(layer_num) = layer_str.parse::<usize>() {
                        max_layer = max_layer.max(layer_num);
                    }
                }
            }
        }

        let num_hidden_layers = max_layer + 1; // Convert to count
        println!("Auto-detected model config: {} layers, {} kv heads", num_hidden_layers, detected_kv_heads);

        let config = SmolVLMConfig {
            num_key_value_heads: detected_kv_heads,
            head_dim: 64,
            num_hidden_layers,
            eos_token_id,
            image_token_id,
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

    fn prompt_split_image(
        image_seq_len: usize,
        image_rows: usize,
        image_cols: usize,
        fake_token_around_image: &str,
        image_token: &str,
        global_image_token: &str,
    ) -> String {
        let mut text_split_images = String::new();
        for n_h in 0..image_rows {
            for n_w in 0..image_cols {
                text_split_images.push_str(&format!(
                    "{}<row_{}_col_{}>{}",
                    fake_token_around_image,
                    n_h + 1,
                    n_w + 1,
                    image_token.repeat(image_seq_len)
                ));
            }
            text_split_images.push('\n');
        }
        text_split_images.push_str(&format!(
            "\n{}{}{}{}",
            fake_token_around_image,
            global_image_token,
            image_token.repeat(image_seq_len),
            fake_token_around_image
        ));
        text_split_images
    }

    fn get_image_prompt_string(
        image_rows: usize,
        image_cols: usize,
        image_seq_len: usize,
        fake_token_around_image: &str,
        image_token: &str,
        global_image_token: &str,
    ) -> String {
        if image_rows == 0 && image_cols == 0 {
            format!(
                "{}{}{}{}",
                fake_token_around_image,
                global_image_token,
                image_token.repeat(image_seq_len),
                fake_token_around_image
            )
        } else {
            SmolVLM::prompt_split_image(
                image_seq_len,
                image_rows,
                image_cols,
                fake_token_around_image,
                image_token,
                global_image_token,
            )
        }
    }

    fn process_vision(
        &self,
        text: &str,
        image: DynamicImage,
    ) -> Result<(String, (Array5<f32>, Array4<i64>))> {
        // Process the image
        let (processed_image, pixel_attention_mask) = self.image_processor.preprocess(image)?;

        // Use 4x4 grid (matching our image processor)
        let image_rows = 4;
        let image_cols = 4;
        let image_seq_len = 64; // SmolVLM2 formula: ((512 // 16) ** 2) / (4**2) = 64

        println!("Image dimensions: {}x{}", processed_image.shape()[3], processed_image.shape()[4]);
        println!("Pixel attention mask shape: {:?}", pixel_attention_mask.shape());

        // Get the image prompt string
        let image_prompt = SmolVLM::get_image_prompt_string(
            image_rows,
            image_cols,
            image_seq_len,
            "<fake_token_around_image>",
            "<image>",
            "<global-img>",
        );

        // Replace the <image> token in the text with our expanded prompt
        let prompt = text.replace("<image>", &image_prompt);

        Ok((prompt, (processed_image, pixel_attention_mask)))
    }

    fn generate(&self, prompt: &str, image: DynamicImage) -> Result<String> {
        // Process the image and get the expanded prompt
        let (expanded_prompt, (processed_image, pixel_attention_mask)) = self.process_vision(prompt, image)?;

        // Tokenize the expanded prompt
        let encoding = self.processor.encode(expanded_prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Convert attention mask to Vec
        let attention_mask = attention_mask.iter().map(|&x| x as u32).collect::<Vec<_>>();
        let input_ids = input_ids.iter().map(|&x| x as u32).collect::<Vec<_>>();

        // Initialize past key values
        let batch_size = 1;
        let mut past_key_values: HashMap<String, Array<f32, _>> = HashMap::new();
        for layer in 0..self.config.num_hidden_layers {
            let key_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            let value_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            past_key_values.insert(format!("past_key_values.{}.key", layer), key_array);
            past_key_values.insert(format!("past_key_values.{}.value", layer), value_array);
        }

        // Get image features - compute once and store
        let mut vision_inputs: HashMap<&str, Value> = HashMap::new();
        vision_inputs.insert("pixel_values", Value::from_array(processed_image.clone())?.into());

        // Convert attention mask to boolean while maintaining shape (batch, num_frames, height, width)
        let pixel_attention_mask_bool = pixel_attention_mask.map(|&x| x != 0);
        vision_inputs.insert("pixel_attention_mask", Value::from_array(pixel_attention_mask_bool.clone())?.into());

        let vision_outputs = self.vision_session.run(vision_inputs)?;
        let image_features = vision_outputs[0].try_extract_tensor::<f32>()?.to_owned();
        println!("Image features shape: {:?}", image_features.shape());

        // Calculate total size for first dimension (17 * 64 = 1088)
        let total_size = image_features.shape()[0] * image_features.shape()[1];
        let feature_dim = image_features.shape()[2]; // Should be 576 based on output
        let image_features_reshaped = image_features.into_shape_with_order((total_size, feature_dim))?;
        println!("Reshaped features shape: {:?}", image_features_reshaped.shape());

        // Generation loop
        let max_new_tokens = 1024;
        let mut generated_tokens = Vec::new();
        let mut input_ids = Array::from_vec(input_ids.iter().map(|&x| x as i64).collect())
            .into_shape_with_order((1, input_ids.len()))?
            .into_owned();
        let mut attention_mask = Array::from_vec(attention_mask.iter().map(|&x| x as i64).collect())
            .into_shape_with_order((1, attention_mask.len()))?
            .into_owned();

        // Calculate cumulative position_ids like in Python: np.cumsum(attention_mask, axis=-1)
        let mut position_ids_vec = Vec::new();
        let mut cumsum = 0i64;
        for &mask_val in attention_mask.iter() {
            cumsum += mask_val;
            position_ids_vec.push(cumsum);
        }
        let mut position_ids = Array::from_vec(position_ids_vec)
            .into_shape_with_order((1, attention_mask.len()))?
            .into_owned();

        for _ in 0..max_new_tokens {
            // Get input embeddings
            let mut embed_inputs: HashMap<&str, Value> = HashMap::new();
            embed_inputs.insert("input_ids", Value::from_array(input_ids.clone())?.into());
            let embed_outputs = self.embed_session.run(embed_inputs)?;
            let mut input_embeds = embed_outputs[0].try_extract_tensor::<f32>()?.to_owned();

            // Replace image token embeddings with image features
            let mut feature_idx = 0;
            for i in 0..input_ids.shape()[1] {
                if input_ids[[0, i]] == self.config.image_token_id as i64 {
                    let mut slice = input_embeds.slice_mut(ndarray::s![0, i, ..]);
                    slice.assign(&image_features_reshaped.slice(ndarray::s![feature_idx, ..]));
                    feature_idx += 1;
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

            // Get next token from last position - use argmax like in Python
            let last_idx = logits.shape()[1] - 1;
            let logits_slice = logits.slice(ndarray::s![0, last_idx, ..]);

            // Use argmax to get the most likely token (like Python: logits[:, -1].argmax(-1, keepdims=True))
            let next_token = logits_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap();

            // Update inputs for next iteration - following Python logic exactly
            input_ids = Array::from_vec(vec![next_token]).into_shape_with_order((1, 1))?;

            // Concatenate new attention mask (Python: attention_mask = np.ones_like(input_ids))
            let new_attention = Array::<i64, _>::ones((1, 1));
            let total_length = attention_mask.len() + new_attention.len();
            let mut attention_vec = attention_mask.iter().copied().collect::<Vec<_>>();
            attention_vec.extend(new_attention.iter().copied());
            attention_mask = Array::from_vec(attention_vec).into_shape_with_order((1, total_length))?;

            // Update position_ids (Python: position_ids = position_ids[:, -1:] + 1)
            let current_pos = position_ids[[0, position_ids.shape()[1] - 1]] + 1;
            position_ids = Array::from_vec(vec![current_pos]).into_shape_with_order((1, 1))?;

            // Update past key values - decoder outputs are: [logits, past_key_0, past_value_0, past_key_1, past_value_1, ...]
            for i in 0..self.config.num_hidden_layers {
                let key = format!("past_key_values.{}.key", i);
                let value = format!("past_key_values.{}.value", i);

                if let Some(past_key) = past_key_values.get_mut(&key) {
                    if i * 2 + 1 < decoder_outputs.len() {
                        let present_key = decoder_outputs[i * 2 + 1].try_extract_tensor::<f32>()?.to_owned();
                        *past_key = present_key.into_dyn();
                    }
                }

                if let Some(past_value) = past_key_values.get_mut(&value) {
                    if i * 2 + 2 < decoder_outputs.len() {
                        let present_value = decoder_outputs[i * 2 + 2].try_extract_tensor::<f32>()?.to_owned();
                        *past_value = present_value.into_dyn();
                    }
                }
            }

            // Add to generated tokens
            generated_tokens.push(next_token as u32);

            // Check for EOS token
            if next_token == self.config.eos_token_id as i64 {
                println!("\n[EOS token detected, stopping generation]");
                break;
            }

            // Print token as we generate (streaming) - matching Python behavior
            if let Ok(token_str) = self.processor.decode(&[next_token as u32], true) {
                print!("{}", token_str);
                std::io::stdout().flush().ok(); // Ensure immediate output
            }
        }
        println!();

        // Decode the generated tokens
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