import tensorflow as tf
import streamlit as st
import numpy as np

from tensorflow import keras
from keras.models import load_model

model = load_model('vegetable_and_fruits_classification_model.keras')

data_category =[
    'apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon'
]

image_width = 180
image_height = 180

st.header('Vegetable and Fruits Classification')
img = st.text_input('Enter Image Name:', 'apple.jpg')

image_load = tf.keras.utils.load_img(img, target_size=(image_width, image_height))
image_arr = tf.keras.utils.array_to_img(image_load)
img_batch = tf.expand_dims(image_arr, axis=0)

predict = model.predict(img_batch)
score = tf.nn.softmax(predict)

st.image(img)
st.write('Image is {} with {:.2f} percent confidence'.format(data_category[np.argmax(score)], 100 * np.max(score)))


