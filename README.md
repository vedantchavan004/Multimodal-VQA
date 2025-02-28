# Multimodal Visual Question Answering (VQA) Project with BLIP-2

## 📚 **Project Overview**
The **Multimodal Visual Question Answering (VQA) Project** utilizes the **BLIP-2 (Bootstrapping Language-Image Pre-training)** model to answer natural language questions about images. The project combines **image analysis** with **natural language processing (NLP)** to generate meaningful responses to visual inputs. This is particularly useful for applications like **image captioning**, **interactive AI assistants**, and **assistive technologies**.

## 🚀 **Features**
- **Multimodal Input:** Combines images and text for advanced AI interactions.
- **Advanced Model:** Uses **BLIP-2 (Salesforce/blip2-flan-t5-xl)** for high-quality image-to-text generation.
- **GPU Acceleration:** Optimized for **NVIDIA A100 GPUs**, but also works on CPU.
- **Easy Integration:** The code is modular and easy to integrate into other applications.

## 📂 **Dataset**
The project uses a **custom image** for testing, eliminating the need for large datasets. However, it can be extended to use datasets like **COCO**, **Flickr30k**, or **Visual Genome** for large-scale applications.

## 🛠️ **Technologies Used**
- **Python:** Programming language.
- **PyTorch:** Deep learning framework.
- **Hugging Face Transformers:** For the **BLIP-2 model**.
- **Matplotlib:** To display images and results.
- **PIL (Python Imaging Library):** For image processing.

## 🧠 **Model Architecture**
- **Image Processing:** Utilizes the **BLIP-2 processor** to preprocess images.
- **Visual-Language Fusion:** The **BLIP-2 model** combines **image embeddings** with **textual inputs**.
- **Text Generation:** Generates **natural language responses** to image-based questions.

## ⚙️ **Installation**
```sh
# Clone the repository
git clone https://github.com/vedantchavan004/Multimodal-VQA.git
cd vqa-blip2-project

# Install dependencies
pip install -r requirements.txt
```

### **Dependencies:**
- `torch`
- `transformers`
- `matplotlib`
- `PIL`

## 🚦 **How to Run**
```sh
# Run the project
python main.py
```

1. **Ensure the image file** is available at the specified path.
2. **Update the question** in `main.py` if needed.

## 📝 **Example Output**
```sh
Using device: cuda
Question: What is in the image?
Answer: A person sitting with a dog against a scenic sunset.
```

## 🖼️ **Sample Image**
![Sample Image](path/to/sample_image.jpg)

## 🎯 **Future Enhancements**
- **Dataset Integration:** Connect to larger datasets for more robust testing.
- **Web Interface:** Develop a **Streamlit** or **Flask** app for an interactive UI.
- **Model Fine-Tuning:** Enhance performance on domain-specific tasks.

## 🤝 **Contributing**
Feel free to **fork** this project and **submit pull requests**. Suggestions and contributions are always welcome!

## 📄 **License**
This project is licensed under the **MIT License**.

## 🙏 **Acknowledgements**
- [Hugging Face](https://huggingface.co/) for the **BLIP-2 model**.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [Matplotlib](https://matplotlib.org/) and [PIL](https://pillow.readthedocs.io/) for visualization.

---

