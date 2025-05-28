# quant_prune_blip1_demo.py

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time
from torch.nn.utils import prune
import copy

# --- Configuration ---
MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_PATH = "stock-photo-159533631.jpg"
MAX_LENGTH = 50
REPEATS = 3

# --- Utility Functions ---
def measure_time(func, *args, repeats=REPEATS):
    """Measures average execution time of func over a number of repeats"""
    timings = []
    for _ in range(repeats):
        start = time.time()
        _ = func(*args)
        timings.append(time.time() - start)
    return sum(timings) / len(timings)

# --- Load Model & Processor ---
def load_model():
    """Loads the BLIP-1 model and processor onto CPU"""
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).eval().to("cpu")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    return model, processor

# --- Inference Function ---
def infer_caption(model, processor, image_path):
    """Runs inference to generate a caption for the given image"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=MAX_LENGTH)
    caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# --- Pruning Function ---
def prune_model(model, amount=0.3):
    """Applies L1 unstructured pruning to all Linear layers"""
    pruned = copy.deepcopy(model)
    for module in pruned.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return pruned

# --- Main Demo ---
def main():
    # Load baseline model
    model, processor = load_model()

    # Baseline CPU inference
    baseline_time = measure_time(infer_caption, model, processor, IMAGE_PATH)
    baseline_caption = infer_caption(model, processor, IMAGE_PATH)
    print(f"Baseline Caption: {baseline_caption}")
    print(f"Baseline CPU time: {baseline_time:.2f} sec")

    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quant_time = measure_time(infer_caption, quantized_model, processor, IMAGE_PATH)
    quant_caption = infer_caption(quantized_model, processor, IMAGE_PATH)
    print(f"Quantized Caption: {quant_caption}")
    print(f"Quantized CPU time: {quant_time:.2f} sec")

    # Pruning + Quantization
    pruned_model = prune_model(model, amount=0.3)
    pruned_quant_model = torch.quantization.quantize_dynamic(
        pruned_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    pruned_time = measure_time(infer_caption, pruned_quant_model, processor, IMAGE_PATH)
    pruned_caption = infer_caption(pruned_quant_model, processor, IMAGE_PATH)
    print(f"Pruned+Quant Caption: {pruned_caption}")
    print(f"Pruned+Quant CPU time: {pruned_time:.2f} sec")

    # Summary of speedups
    speedup_quant = baseline_time / quant_time if quant_time else float('inf')
    speedup_prune = baseline_time / pruned_time if pruned_time else float('inf')
    print(f"Speedup (Quantization): {speedup_quant:.2f}x")
    print(f"Speedup (Prune+Quant): {speedup_prune:.2f}x")

if __name__ == "__main__":
    main()
