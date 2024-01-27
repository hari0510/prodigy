import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load the model and labels
model = load_model("keras_model_food.h5", compile=False)
labels = [label.strip() for label in open("labels_food.txt", "r").readlines()]

# Placeholder calorie values (replace with actual values)
calorie_values = {
    "chicken_razala": 300,
    "imarti": 150,
    "jalebi": 200,
    "Kachori": 250,
    "Kadai_Paneer": 400,
    "Kadhi_Pakoda": 300,
    "Kajjjkaya": 180,
    "Kakhinada_Kajha": 220,
    "kalakhand": 180,
    "Karela_Bharta": 120,
    "Kofta": 350,
    "Paniyaram": 160,
    "Lazhi": 200,
    "Ledi_Keni": 250,
}

def preprocess_image(img_file):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    return data

def fetch_calories(food_class):
    return f"Calories: {calorie_values.get(food_class, 'N/A')} (per 100 grams)"

def set_custom_style():
    # Set a custom background color and text style
    st.markdown(
        """
        <style>
            body {
                background-color: #000000;  /* Black background color */
                color: #FFFFFF;  /* White text color */
                font-family: 'Times New Roman', Times, serif;  /* Times New Roman font */
            }
            .css-19ih76x h1 {
                color: #FF6347;  /* Tomato heading color */
                text-align: center;
                font-size: 36px;
            }
            .css-19ih76x p {
                font-size: 18px;
                text-align: center;
                margin-top: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def run():
    set_custom_style()
    st.title("Image Classification With Calorie")
    st.markdown("Upload image!")

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    
    if img_file is not None:
        st.image(img_file, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        data = preprocess_image(img_file)

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = labels[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        st.success(f"Predicted: {class_name[2:]} (Confidence Score: {confidence_score:.2%})")
        cal = fetch_calories(class_name[2:])
        
        if cal:
            st.warning(cal)
            st.info("Note: Calorie values are approximate and may vary.")
            st.info("For accurate nutritional information, consult a nutritionist or use a food database.")
            st.info("Enjoy your meal!")

if __name__ == "__main__":
    run()
