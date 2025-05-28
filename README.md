# BLIP-1 Optimization for Efficient Image Captioning

This repository demonstrates advanced model optimization techniques on the **BLIP-1** image-captioning model (`Salesforce/blip-image-captioning-base`). It provides:

* **Baseline inference** on CPU
* **Dynamic quantization** of Linear layers
* **Unstructured pruning** (L1) combined with quantization
* **Benchmark results** comparing speedups and validating accuracy

## Repository Structure

```plaintext
blip1-opt-demo/
├── quant_prune_blip1_demo.py   # Main demo script
├── stock-photo-159533631.jpg   # Example image used for captions
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
3. **Place your test image**

   * here I have `stock-photo-159533631.jpg` in the repo root.

## Demo Script: `quant_prune_blip1_demo.py`

This script executes the following steps:

1. **Baseline CPU inference**:

   * Loads BLIP-1 model and processor
   * Generates a caption for `stock-photo-159533631.jpg`
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

# Results Interpretation

Below is an interpretation of the benchmark outcomes for the BLIP-1 model optimization demo.

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

## Key Insights

* **Dynamic Quantization Effectiveness**: Applying PyTorch dynamic quantization to all `torch.nn.Linear` layers reduced the model’s average CPU inference time from **0.69 s** to **0.48 s**, a **1.42×** speedup. Quantization compresses weight precision from 32-bit to 8-bit, cutting memory bandwidth usage and improving cache locality.

* **Pruning Impact**: Unstructured L1 weight pruning at 30% sparsity followed by dynamic quantization achieved **0.49 s** (1.40× speedup). Because PyTorch’s default CPU backend does not exploit unstructured sparsity, pruned weights still participate in dense operations, so quantization remains the primary driver of acceleration.

* **Accuracy Preservation**: All three model variants produced the same caption (`"a man and his dog"`), demonstrating that these optimizations did not degrade the output quality for this example.

* **Optimization Trade-offs**:

  * Dynamic quantization provides a **quick, impactful** improvement for transformer-based vision-language models on CPU.
  * Unstructured pruning alone does **not** yield additional speedup without a specialized sparse inference engine.
  * For further performance, consider:

    * **Structured pruning** (e.g., removing entire neurons or attention heads) to reduce compute complexity.
    * **Sparse inference libraries** (e.g., TVM, PyTorch Sparse) that can leverage weight sparsity at runtime.
    * **TensorRT** or **ONNX Runtime** INT8 quantization for GPU and cross-platform deployment.

## Conclusion

Dynamic quantization is a reliable, easily applied optimization for CPU deployment of large vision-language models, delivering substantial speedup with minimal code changes and zero accuracy loss. Advanced techniques like structured pruning and sparse inference are promising next steps to push performance further.

## Results Interpretation

Below is an interpretation of the benchmark outcomes for the BLIP-1 model optimization demo:

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

**Key Insights**:

* **Dynamic Quantization** reduced CPU inference time by **1.42×** with no loss in caption accuracy.
* **Pruning + Quantization** achieved **1.40×** speedup; pruning alone doesn’t speed up dense compute.
* **Accuracy Preservation**: All variants yielded the correct caption.

For deeper optimization, consider:

* **Structured pruning** to remove neurons or heads.
* **Sparse inference libraries** to leverage weight sparsity.
* **ONNX Runtime INT8 quantization** for cross-platform deployment.

## requirements.txt

```text
torch>=2.0.0
transformers>=4.30.0
Pillow>=8.0.0
```


## Next Steps / Extensions

* **Structured pruning** to remove entire neurons or heads for real compute savings
* **ONNX export & quantization** for cross-platform deployment
* **TensorRT integration** for GPU edge acceleration
* **Beam search or top‑k sampling** to demonstrate decoding optimizations

## License

This project is released under the MIT License.
