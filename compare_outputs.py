import numpy as np

# Load the .npy files
python_pixel_values = np.load("python_pixel_values.npy")
rust_pixel_values = np.load("rust_pixel_values.npy")

python_pixel_attention_mask = np.load("python_pixel_attention_mask.npy")
rust_pixel_attention_mask = np.load("rust_pixel_attention_mask.npy")

# Compare the outputs
print("pixel_values close:", np.allclose(python_pixel_values, rust_pixel_values))
print("pixel_attention_mask close:", np.allclose(python_pixel_attention_mask, rust_pixel_attention_mask)) 