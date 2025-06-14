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