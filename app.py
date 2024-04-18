import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Preprocess the new image
def preprocess_image(image):
    img = image.resize((125, 100)) 
    img = np.array(img) / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

# Map class labels to class names
class_names = {
    0: "Melanocytic nevi",
    1: "Melanoma",
    2: "Benign keratosis-like lesions",
    3: "Basal cell carcinoma",
    4: "Actinic keratoses",
    5: "Vascular lesions",
    6: "Dermatofibroma"
}

def main():
    st.title('Skin disease Classification')
    st.write('Upload an image for classification')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Load the pre-trained model
        model_path = 'oversampled.keras'  # Update with your model path
        model = load_model(model_path)

        # Make predictions
        predictions = model.predict(preprocessed_image)

        # Get class label
        class_label = np.argmax(predictions)

        # Get class name
        predicted_class_name = class_names[class_label]

        st.write("Predicted class name:", predicted_class_name)

if __name__ == "__main__":
    main()
