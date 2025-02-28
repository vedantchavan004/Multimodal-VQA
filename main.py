# Multimodal Visual Question Answering (VQA) Project with BLIP-2
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import matplotlib.pyplot as plt
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the BLIP-2 model and processor from Hugging Face
model_name = 'Salesforce/blip2-flan-t5-xl'
processor = Blip2Processor.from_pretrained(model_name)
language_model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)

# Load image
image_path = 'stock-photo-159533631.jpg'
img = Image.open(image_path).convert('RGB')

# question for testing
question = 'What is in the image?'

# Prepare the input for the BLIP-2 model
inputs = processor(images=img, text=question, return_tensors='pt').to(device)

# Answer using the BLIP-2 model
with torch.no_grad():
    outputs = language_model.generate(**inputs, max_length=50)
answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f'Question: {question}')
print(f'Answer: {answer}')
plt.imshow(img)
plt.axis('off')
plt.title(f'Q: {question}\nA: {answer}')
plt.show()
