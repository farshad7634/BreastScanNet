import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from model import BreastCancerClassifier
import os

# Initialize model
model = BreastCancerClassifier(num_classes=3)

# Construct the relative path to the model file
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # for grayscale
])

# Class labels
class_names = ['benign', 'malignant', 'normal']

def predict(image_path):
    img = Image.open(image_path).convert('L')  # Grayscale
    img_tensor = transform(img).unsqueeze(0)   # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return class_names[prediction]

# GUI setup
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('L')
        img_resized = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_resized)
        panel.config(image=img_tk)
        panel.image = img_tk

        result = predict(file_path)
        label_result.config(text=f"Prediction: {result.upper()}")

# Main GUI window
window = tk.Tk()
window.title("BreastScanNet Classifier")

panel = tk.Label(window)
panel.pack()

btn_browse = tk.Button(window, text="Select Sonogram Image", command=browse_image)
btn_browse.pack(pady=10)

label_result = tk.Label(window, text="Prediction: N/A", font=("Arial", 14))
label_result.pack(pady=10)

window.mainloop()
