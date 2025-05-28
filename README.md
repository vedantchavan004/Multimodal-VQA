# BLIP-1 Optimization Demo

This repository demonstrates advanced model optimization techniques on the **BLIP-1** image-captioning model (`Salesforce/blip-image-captioning-base`). It provides:

* **Baseline inference** on CPU
* **Dynamic quantization** of Linear layers
* **Unstructured pruning** (L1) combined with quantization
* **Benchmark results** comparing speedups and validating accuracy

## Repository Structure

```plaintext
blip1-opt-demo/
├── quant_prune_blip1_demo.py   # Main demo script
├── sample_image.jpg            # Example image used for captions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup and Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/blip1-opt-demo.git
   cd blip1-opt-demo
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place your test image**

   * Rename or copy an image as `sample_image.jpg` in the repo root.

## Demo Script: `quant_prune_blip1_demo.py`

This script executes the following steps:

1. **Baseline CPU inference**:

   * Loads BLIP-1 model and processor
   * Generates a caption for `sample_image.jpg`
   * Measures average CPU inference time

2. **Dynamic quantization**:

   * Applies PyTorch dynamic quantization to all `torch.nn.Linear` layers
   * Re-runs inference and measures speedup

3. **Unstructured pruning + quantization**:

   * Applies L1 unstructured pruning (30% sparsity) to Linear weights
   * Applies dynamic quantization on the pruned model
   * Re-runs inference and measures speedup

4. **Summary**:

   * Prints captions and timings for each stage
   * Reports speedup factors

## Sample Output

```text
Baseline Caption: a man and his dog
Baseline CPU time: 0.69 sec
Quantized Caption: a man and his dog
Quantized CPU time: 0.48 sec
Pruned+Quant Caption: a man and his dog
Pruned+Quant CPU time: 0.49 sec
Speedup (Quantization): 1.42x
Speedup (Prune+Quant): 1.40x
```

## requirements.txt

```text
torch>=2.0.0
transformers>=4.30.0
Pillow
```

## Next Steps / Extensions

* **Structured pruning** to remove entire neurons or heads for real compute savings
* **ONNX export & quantization** for cross-platform deployment
* **TensorRT integration** for GPU edge acceleration
* **Beam search or top‑k sampling** to demonstrate decoding optimizations

## License

This project is released under the MIT License.
