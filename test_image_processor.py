from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image
import numpy as np

# Load config and processor
model_id = "HuggingFaceTB/SmolVLM2-500M-Instruct"
config = AutoConfig.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Load and process image
image = load_image("boat.png")
inputs = processor(images=[image], return_tensors="np")

# Print shapes and values
print("Python Output:")
print("pixel_values shape:", inputs['pixel_values'].shape)
print("pixel_attention_mask shape:", inputs['pixel_attention_mask'].shape)
print("\npixel_values min:", inputs['pixel_values'].min())
print("pixel_values max:", inputs['pixel_values'].max())
print("pixel_values mean:", inputs['pixel_values'].mean())
print("pixel_values std:", inputs['pixel_values'].std())

# Save the processed image for comparison
np.save("python_pixel_values.npy", inputs['pixel_values'])
np.save("python_pixel_attention_mask.npy", inputs['pixel_attention_mask']) 