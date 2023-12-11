import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#already model is built and saved
model = tf.keras.models.load_model("C:/Users/Vishwa Raj/PycharmProjects/Digit_classification/model/Digit_classification.model")

img = cv2.imread(f"digits_test/digit1.png")[:, :, 0]
img = np.invert(np.array([img]))
plt.matshow(img.reshape(28,28))
plt.show()
prediction = model.predict(img)
print("PREDICT:",prediction)
print("The number you wrote is :", np.argmax(prediction))

# image_number = 1
# while os.path.isfile(f"digits_test/digit{image_number}.png"):
