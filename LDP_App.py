
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import google.generativeai as genai

# Load the trained lung disease detection model
model = load_model(r'D:/LDP Project/Lung Disease Dataset/lung_disease_detection_model.h5')

# Define the labels
labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

# Initialize the Gemini API
GEMINI_API_KEY = "AIzaSyDoaGwqLnuiB9bV26jm5gNLum7DTTYQM1M"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

def preprocess_image(image):
    image = img_to_array(image)
    image = cv2.resize(image, (150,150))  # Resize to match the input shape of the model
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

st.title('Lung Disease Detection and AI Symptom Checker')

# No need to display AI model selection, hardcoding to "GEMINI"
ai_model = "GEMINI" 

# Create tabs for image upload and symptom checker
tab1, tab2 = st.tabs(["X-Ray Scan Analysis", "AI Symptom Checker"])

with tab1:
    st.write("Upload an X-ray scan to predict the lung disease.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        
        # Resize the image to 50x50
        resized_image = image.resize((80, 80))
        
        st.image(resized_image, caption='Uploaded Image', use_column_width=False)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = labels[np.argmax(prediction)]
        
        # Display the prediction
        st.write(f"Prediction: **{predicted_class}**")
        
        if predicted_class != 'Normal':
            ai_prompt = f"Provide a brief description of {predicted_class}, including common symptoms, potential causes, and general advice for patients. Keep the response concise and informative."
            ai_response = get_gemini_response(ai_prompt)
            st.write(ai_response)

with tab2:
    st.write(f"Describe your symptoms to get {ai_model}-powered assistance.")
    user_input = st.text_area("Enter your symptoms and concerns:")
    if st.button("Get AI Assistance"):
        if user_input:
            ai_prompt = f"The user has described the following symptoms and concerns related to lung health: '{user_input}'. Provide a helpful response including possible conditions to consider, general advice, and the importance of consulting a healthcare professional. Keep the response concise and informative."
            
            response = get_gemini_response(ai_prompt)
            
            st.write(f"{ai_model} Assistant:", response)
        
            st.write("Please enter some symptoms or concerns.")

st.warning("This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.")


