import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np

model = load_model("Like_model.h5")
path = "Like Dislike/val/dislike/21.jpg"

image = tf.keras.preprocessing.image.load_img(path, target_size=(200, 200))
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])

prediction = model.predict(input_arr)

if prediction == 0:
    print("Creature on photo:  ", path, "is a dislike")
else:
    print("Creature on photo: ", path, "is a like")