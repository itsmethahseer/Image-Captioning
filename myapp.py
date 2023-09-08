import streamlit as st
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os
from keras.models import Model




vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load the caption generation model and tokenizer
model = load_model('best_model.h5')
tokenizer = joblib.load('my_tokenizer.pkl')
max_length = 35


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text




st.title("Image Captioning Web App")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).resize((224,224))
    
    if st.button("predict caption"):
        # Generate caption using the model
        # image = load_img(image_path, target_size=(224, 224))
        image_p = img_to_array(image)
        image_p = image_p.reshape((1, image_p.shape[0], image_p.shape[1], image_p.shape[2]))
        image_p = preprocess_input(image_p)
        feature = vgg_model.predict(image_p, verbose=0)
        caption = predict_caption(model, feature, tokenizer, max_length)
        st.write("Caption:", caption)
    
    
        # Display the uploaded image
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

   
