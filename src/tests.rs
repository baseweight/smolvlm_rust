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
fn test_boat_image_processing() {
    // Load the boat image
    let img = image::open("boat.png").expect("Failed to load boat image");
    
    // Process the image
    let processor = SmolVLMImageProcessor::new();
    let (pixel_values, pixel_attention_mask) = processor.preprocess(img).unwrap();

    // Load Python-generated values for comparison
    let python_pixel_values: Array5<f32> = read_npy("python_pixel_values.npy").unwrap();
    let python_pixel_attention_mask: Array4<i64> = read_npy("python_pixel_attention_mask.npy").unwrap();

    // Save our results to numpy files for inspection
    write_npy("rust_pixel_values.npy", &pixel_values).unwrap();
    write_npy("rust_pixel_attention_mask.npy", &pixel_attention_mask).unwrap();

    // Verify the shapes match
    assert_eq!(pixel_values.shape(), python_pixel_values.shape(), "Pixel values shape mismatch");
    assert_eq!(pixel_attention_mask.shape(), python_pixel_attention_mask.shape(), "Attention mask shape mismatch");

    // Compare values with Python output
    let max_diff = pixel_values.iter()
        .zip(python_pixel_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    
    println!("Maximum difference between Rust and Python pixel values: {}", max_diff);

    // Print first frame statistics (slice) and first few pixel values (for debugging)
    let binding_rust = pixel_values.index_axis(ndarray::Axis(0), 0);
    let rust_first_frame = binding_rust.index_axis(ndarray::Axis(0), 0);
    let binding_py = python_pixel_values.index_axis(ndarray::Axis(0), 0);
    let py_first_frame = binding_py.index_axis(ndarray::Axis(0), 0);
    println!("\nRust First Frame (slice) statistics:");
    println!("min: {}", rust_first_frame.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
    println!("max: {}", rust_first_frame.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    println!("mean: {}", rust_first_frame.mean().unwrap());
    println!("std: {}", (rust_first_frame.var(1.0)).sqrt());
    println!("\nPython First Frame (slice) statistics:");
    println!("min: {}", py_first_frame.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
    println!("max: {}", py_first_frame.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    println!("mean: {}", py_first_frame.mean().unwrap());
    println!("std: {}", (py_first_frame.var(1.0)).sqrt());
    println!("\nRust First Frame (slice) first few pixel values (channels 0,1,2, positions (0,0)–(2,2)):");
    for c in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                println!("Channel {}, Position ({}, {}): {}", c, y, x, rust_first_frame[[c, y, x]]);
            }
        }
    }
    println!("\nPython First Frame (slice) first few pixel values (channels 0,1,2, positions (0,0)–(2,2)):");
    for c in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                println!("Channel {}, Position ({}, {}): {}", c, y, x, py_first_frame[[c, y, x]]);
            }
        }
    }

    assert!(max_diff < 1e-5, "Pixel values differ too much from Python implementation");

    // Compare attention masks
    let mask_matches = pixel_attention_mask.iter()
        .zip(python_pixel_attention_mask.iter())
        .all(|(a, b)| *a == *b);
    assert!(mask_matches, "Attention masks don't match Python implementation");

    // Print statistics for debugging (overall)
    let min = pixel_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = pixel_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = pixel_values.mean().unwrap();
    let std = (pixel_values.var(1.0)).sqrt();

    println!("\nRust Output (overall):");
    println!("pixel_values shape: {:?}", pixel_values.shape());
    println!("pixel_attention_mask shape: {:?}", pixel_attention_mask.shape());
    println!("\npixel_values min: {}", min);
    println!("pixel_values max: {}", max);
    println!("pixel_values mean: {}", mean);
    println!("pixel_values std: {}", std);

    // Print Python statistics for comparison (overall)
    let py_min = python_pixel_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let py_max = python_pixel_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let py_mean = python_pixel_values.mean().unwrap();
    let py_std = (python_pixel_values.var(1.0)).sqrt();

    println!("\nPython Output (overall):");
    println!("pixel_values min: {}", py_min);
    println!("pixel_values max: {}", py_max);
    println!("pixel_values mean: {}", py_mean);
    println!("pixel_values std: {}", py_std);
} 