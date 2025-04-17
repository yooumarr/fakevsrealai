import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Conv2d(in_channels=3, out_channels= 6, kernel_size=(5,5),stride=3)
        self.r1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=1)
        self.l2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3,3), stride=1)
        self.r2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=1)
        self.lin1 = nn.Linear(192,100)
        self.lin2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.l1(x)
        x = self.r1(x)
        x = self.maxpool1(x)
        x = self.l2(x)
        x = self.r2(x)
        x = self.maxpool2(x)
        x = x.flatten(1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x

# ---------------------- Streamlit Integration ----------------------
st.title("Real vs AI Image Detector")

# Load the trained model
loaded_model = Net()
try:
    loaded_model.load_state_dict(torch.load(r'real_vs_ai_model.pth', map_location=torch.device('cpu')))
    loaded_model.eval()
except FileNotFoundError:
    st.error("Error: 'real_vs_ai_model.pth' not found. Make sure it's in the same directory.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

def preprocess_single_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Ensure consistent input size
        transforms.ToTensor()
    ])
    img_t = transform(image).unsqueeze(0)
    return img_t

def predict(image_tensor):
    with torch.no_grad():
        output = loaded_model(image_tensor)
        prediction = output.item()
        return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_container_width =True)

    preprocessed_image = preprocess_single_image(image)
    prediction_value = predict(preprocessed_image)

    st.subheader("Prediction:")
    if prediction_value >= 0.4:
        st.write(f"This image is likely **REAL** (Confidence: {prediction_value:.4f})")
    else:
        st.write(f"This image is likely **AI-GENERATED** (Confidence: {1 - prediction_value:.4f})")
