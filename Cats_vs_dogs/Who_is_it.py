import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np

model = load_model("Cats_Dogs_model.h5")
path = "Dataset/test12/137.jpg"

image = tf.keras.preprocessing.image.load_img(path, target_size=(150, 200))
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])

prediction = model.predict(input_arr)

if prediction == 0:
    print("Creature on photo:  ", path, "is a cat")
else:
    print("Creature on photo: ", path, "is a dog")