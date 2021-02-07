import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

# while True:
#     ret, img = image.read()
#
#     faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5, minSize=(20,20))
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img,(x, y), (x+w, y+h), (255, 0, 0), 2)
#
#     cv2.imshow('VideoCamera', img)
#
#     k = cv2.waitKey(30) & 0xFF
#     if k == 27:
#         break

# NN = AI.neuralNetwork(307200, 1000, 2, 0.3)
#
# for result in range(0, 2):
#     for number in range(1, 11):
#         target = []
#         if result == 0:
#             target = [0.99, 0.01]
#         else:
#             target = [0.01, 0.99]
#
#         image = cv2.imread(f'Dataset/{result}_{number}.jpg')
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image = numpy.asfarray(image)
#         current_image = []
#         for i in range(len(image)):
#             current_image.extend((numpy.asfarray(image[i]) / 255.0 * 0.99) + 0.01)
#         NN.train(current_image, target)
#         print("Trained on ", result, '_', number)
#

batch_size = 2
train_dir = 'Like Dislike/train'
val_dir = 'Like Dislike/val'
test_dir = 'Like Dislike/test'
input_shape = (200, 200, 3)
model_name = 'Like_model.h5'

# Архитектура модели
model = keras.Sequential([
 Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
 MaxPooling2D((2, 2), strides=2),
 Conv2D(16, (3, 3), padding='same', activation='relu'),
 Dropout(0.25),
 MaxPooling2D((2, 2), strides=2),
 Dropout(0.25),
 Conv2D(8, (3, 3), padding='same', activation='relu'),
 MaxPooling2D((2, 2), strides=2),
 Flatten(),
 Dropout(0.3),
 Dense(512, activation='relu'),
 Dense(256, activation='relu'),
 Dense(1,  activation='sigmoid')
])

model.compile(optimizer='adam',
          loss='binary_crossentropy',
          metrics=['accuracy'])


# Инициализация генераторов
datageden = ImageDataGenerator(rescale=1./255)

train_generator = datageden.flow_from_directory(
 train_dir,
 target_size=(200, 200),
 batch_size=batch_size,
 class_mode='binary'
)

val_generator = datageden.flow_from_directory(
 val_dir,
 target_size=(200, 200),
 batch_size=batch_size,
 class_mode='binary'
)

test_generator = datageden.flow_from_directory(
    test_dir,
    target_size=(200, 200),
    batch_size=batch_size,
    class_mode='binary'
)

# Тренировка
model.fit_generator(
 train_generator,
 steps_per_epoch=39 // batch_size,
 epochs=10,
 validation_data=val_generator,
 validation_steps=6 // batch_size
)

model.save(model_name)
scores = model.evaluate_generator(test_generator)
print("Примерная ошибка равна: ", scores[1] * 100 )
