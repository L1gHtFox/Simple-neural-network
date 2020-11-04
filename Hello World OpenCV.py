import cv2
import numpy
import Neural_network as AI

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

NN = AI.neuralNetwork(307200, 1000, 2, 0.3)

for result in range(0, 2):
    for number in range(1, 11):
        target = []
        if result == 0:
            target = [0.99, 0.01]
        else:
            target = [0.01, 0.99]

        image = cv2.imread(f'Dataset/{result}_{number}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = numpy.asfarray(image)
        current_image = []
        for i in range(len(image)):
            current_image.extend((numpy.asfarray(image[i]) / 255.0 * 0.99) + 0.01)
        NN.train(current_image, target)
        print("Trained on ", result, '_', number)

image = cv2.imread(f'Dataset/0_11.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = numpy.asfarray(image)
current_image = []
for i in range(len(image)):
    current_image.extend((numpy.asfarray(image[i]) / 255.0 * 0.99) + 0.01)

print(NN.query(current_image))

# image = cv2.imread('Dataset/0_1.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# while True:
#     cv2.imshow('VideoCamera', image)
#
#     k = cv2.waitKey(30) & 0xFF
#     if k == 27:
#         break