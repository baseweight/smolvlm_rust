use crate::image_processor::SmolVLMImageProcessor;
use image::{DynamicImage, Rgb};
use ndarray::{Array4, Array5};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::fs::File;
use std::io::Write;
use ndarray_npy::{read_npy, write_npy};

#[test]
fn test_image_processor() {
    // Create a test image (512x512 RGB)
    let mut img = image::ImageBuffer::new(512, 512);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([x as u8, y as u8, 255]);
    }
    let img = DynamicImage::ImageRgb8(img);

    // Test image processor
    let processor = SmolVLMImageProcessor::new();
    let (processed, mask) = processor.preprocess(img).unwrap();

    // Check tensor shapes - should always be 17 frames (4x4 grid plus original)
    assert_eq!(processed.shape(), &[1, 17, 3, 512, 512]);
    assert_eq!(mask.shape(), &[1, 17, 512, 512]);

    // Check value ranges
    for val in processed.iter() {
        assert!(!val.is_nan());
        assert!(!val.is_infinite());
    }

    // Check mask values
    for val in mask.iter() {
        assert_eq!(*val, 1);
    }
}

#[test]
fn test_tokenizer() {
    // Create a test prompt
    let prompt = "Can you describe this image?";
    let messages = format!(
        r#"<|im_start|>user
<|im_start|>image<|im_end|>
<|im_start|>text
{prompt}<|im_end|>
<|im_start|>assistant
"#
    );

    // Load tokenizer
    let tokenizer_path = PathBuf::from("tokenizer.json");
    let processor = Tokenizer::from_file(tokenizer_path).unwrap();

    // Test tokenization
    let encoding = processor.encode(messages, true).unwrap();
    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();

    // Check that we got tokens
    assert!(!input_ids.is_empty());
    assert!(!attention_mask.is_empty());

    // Check that attention mask matches input length
    assert_eq!(input_ids.len(), attention_mask.len());

    // Check that all attention mask values are 1
    for &val in attention_mask {
        assert_eq!(val, 1);
    }

    // Test decoding
    let decoded = processor.decode(input_ids, true).unwrap();
    assert!(!decoded.is_empty());
}

#[test]
fn test_image_processor_with_large_image() {
    // Create a large test image (1024x1024 RGB)
    let mut img = image::ImageBuffer::new(1024, 1024);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([x as u8, y as u8, 255]);
    }
    let img = DynamicImage::ImageRgb8(img);

    // Test image processor
    let processor = SmolVLMImageProcessor::new();
    let (processed, mask) = processor.preprocess(img).unwrap();

    // Check tensor shapes - should always be 17 frames (4x4 grid plus original)
    assert_eq!(processed.shape(), &[1, 17, 3, 512, 512]);
    assert_eq!(mask.shape(), &[1, 17, 512, 512]);

    // Check value ranges
    for val in processed.iter() {
        assert!(!val.is_nan());
        assert!(!val.is_infinite());
    }

    // Check mask values
    for val in mask.iter() {
        assert_eq!(*val, 1);
    }
}

#[test]
fn test_prompt_generation() {
    // Test parameters
    let image_rows = 4;
    let image_cols = 4;
    let image_seq_len = 169;
    let fake_token = "<fake_token_around_image>";
    let image_token = "<image>";
    let global_image_token = "<global-img>";

    // Generate the prompt
    let prompt = crate::SmolVLM::get_image_prompt_string(
        image_rows,
        image_cols,
        image_seq_len,
        fake_token,
        image_token,
        global_image_token,
    );

    // Debug: Print the actual prompt
    println!("\nActual prompt:");
    println!("{}", prompt);
    println!("\nNumber of lines: {}", prompt.lines().count());
    println!("\nLines:");
    for (i, line) in prompt.lines().enumerate() {
        println!("Line {}: {}", i + 1, line);
    }

    // Expected format for each row
    let expected_row_format = format!(
        "{}{}<row_{}_col_{}>{}{}",
        fake_token,
        "<row_",
        "{row}",
        "{col}",
        ">",
        image_token.repeat(image_seq_len)
    );

    // Verify each row in the grid
    let lines: Vec<&str> = prompt.lines().collect();
    assert_eq!(lines.len(), 5, "Expected 5 lines (4 grid rows + 1 global image line), got {}", lines.len());

    // Check each grid row
    for row in 0..4 {
        let line = lines[row];
        for col in 0..4 {
            let expected = expected_row_format
                .replace("{row}", &(row + 1).to_string())
                .replace("{col}", &(col + 1).to_string());
            assert!(line.contains(&expected), "Row {} col {} not found in line: {}", row + 1, col + 1, line);
        }
    }

    // Check global image line
    let global_line = lines[4];
    let expected_global = format!(
        "\n{}{}{}{}",
        fake_token,
        global_image_token,
        image_token.repeat(image_seq_len),
        fake_token
    );
    assert_eq!(global_line, expected_global.trim());

    // Verify total number of image tokens
    let total_image_tokens = prompt.matches(image_token).count();
    let expected_total = (image_rows * image_cols + 1) * image_seq_len; // +1 for global image
    assert_eq!(total_image_tokens, expected_total, 
        "Expected {} image tokens but found {}", expected_total, total_image_tokens);
}

#[test]
fn test_token_count_for_example_prompt() {
    let tokenizer_path = std::path::PathBuf::from("tokenizer.json");
    let processor = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();

    // Use the same expansion logic as the main code
    let image_rows = 4;
    let image_cols = 4;
    let image_seq_len = 64;
    let fake_token = "<fake_token_around_image>";
    let image_token = "<image>";
    let global_image_token = "<global-img>";

    let image_prompt = crate::SmolVLM::get_image_prompt_string(
        image_rows,
        image_cols,
        image_seq_len,
        fake_token,
        image_token,
        global_image_token,
    );

    // Use the simpler prompt structure from Python
    let prompt = format!(
        "<|im_start|>User:<image>Can you describe this image?<end_of_utterance>\nAssistant:"
    ).replace("<image>", &image_prompt);

    // Debug output
    println!("\nPrompt before tokenization:");
    println!("{}", prompt);
    let num_image_tokens = prompt.matches("<image>").count();
    println!("\nNumber of <image> tokens in prompt: {}", num_image_tokens);

    let encoding = processor.encode(prompt, true).unwrap();
    let input_ids = encoding.get_ids();
    let num_tokens = input_ids.len();
    println!("\nToken count: {}", num_tokens);
    
    // Print unique tokens and their counts
    let mut token_counts = std::collections::HashMap::new();
    for &id in input_ids {
        *token_counts.entry(id).or_insert(0) += 1;
    }
    println!("\nToken distribution:");
    for (id, count) in token_counts {
        if let Ok(token) = processor.decode(&[id], true) {
            println!("Token '{}' (ID: {}): {} occurrences", token, id, count);
        }
    }

    // Assert the number of <image> tokens is correct
    assert_eq!(num_image_tokens, 1088, "Expected 1088 <image> tokens, got {}", num_image_tokens);
    // Optionally, assert the total token count if you know it
    // assert_eq!(num_tokens, EXPECTED_TOTAL, "Expected {{}} tokens, got {{}}", EXPECTED_TOTAL, num_tokens);
}

#[test]
fn test_tokenizer_image_token_count() {
    use tokenizers::Tokenizer;
    let tokenizer = Tokenizer::from_file("tokenizer.json").expect("Failed to load tokenizer");
    let image_token = "<image>";
    let image_token_id = tokenizer.token_to_id(image_token).expect("No image token ID");
    let image_seq_len = 64;
    let prompt = image_token.repeat(image_seq_len);
    let encoding = tokenizer.encode(prompt, true).expect("Tokenization failed");
    let ids = encoding.get_ids();
    println!("Token IDs: {:?}", ids);
    for (i, &id) in ids.iter().enumerate() {
        let decoded = tokenizer.decode(&[id], true).unwrap_or_else(|_| "<decode error>".to_string());
        println!("Token {}: ID {}: '{}" , i, id, decoded);
    }
    assert_eq!(ids.len(), image_seq_len, "Expected {} tokens, got {}", image_seq_len, ids.len());
    assert!(ids.iter().all(|&id| id == image_token_id), "Not all tokens are <image> token ID");
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    const STATS_TOLERANCE: f32 = 0.01; // 1% tolerance for statistics comparison

    fn assert_within_tolerance(actual: f32, expected: f32, name: &str) {
        let diff = (actual - expected).abs();
        let tolerance = expected.abs() * STATS_TOLERANCE;
        assert!(
            diff <= tolerance,
            "{}: expected {} but got {} (diff: {}, tolerance: {})",
            name,
            expected,
            actual,
            diff,
            tolerance
        );
    }

    #[test]
    fn test_boat_image_processing() -> Result<()> {
        let processor = SmolVLMImageProcessor::new();
        let image = image::open("boat.png")?;
        let (pixel_values, pixel_attention_mask) = processor.preprocess(image)?;

        // Calculate statistics for comparison with JS
        let data = pixel_values.as_slice().unwrap();
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();

        println!("\nRust Statistics:");
        println!("Shape: {:?}", pixel_values.shape());
        println!("Min: {}", min);
        println!("Max: {}", max);
        println!("Mean: {}", mean);
        println!("Std: {}", std);

        // Load JavaScript-generated values for comparison
        let js_values = std::fs::read_to_string("js_pixel_values.txt")?;
        let js_stats = js_values.lines()
            .nth(1)
            .and_then(|line| line.strip_prefix("stats:"))
            .ok_or_else(|| anyhow::anyhow!("Failed to parse JS stats"))?;

        // Parse the stats manually since we don't have serde_json
        let js_stats = js_stats.trim_matches(|c| c == '{' || c == '}');
        let mut js_min = 0.0;
        let mut js_max = 0.0;
        let mut js_mean = 0.0;
        let mut js_std = 0.0;

        for pair in js_stats.split(',') {
            let (key, value) = pair.split_once(':').unwrap();
            let value = value.trim().parse::<f32>().unwrap();
            match key.trim().trim_matches('"') {
                "min" => js_min = value,
                "max" => js_max = value,
                "mean" => js_mean = value,
                "std" => js_std = value,
                _ => {}
            }
        }

        println!("\nJavaScript Statistics:");
        println!("Min: {}", js_min);
        println!("Max: {}", js_max);
        println!("Mean: {}", js_mean);
        println!("Std: {}", js_std);

        // Compare with tolerance
        assert_within_tolerance(min, js_min, "min");
        assert_within_tolerance(max, js_max, "max");
        assert_within_tolerance(mean, js_mean, "mean");
        assert_within_tolerance(std, js_std, "std");

        Ok(())
    }
} 