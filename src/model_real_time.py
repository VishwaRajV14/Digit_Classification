import cv2
import numpy as np
import tensorflow as tf
import pickle
#########################
width = 200
height = 200
#########################


#camera set up
cap = cv2.VideoCapture(2)
cap.set(3,width)
cap.set(4,height)

#load the model
# pickle_in = open("C:/Users/Vishwa Raj/PycharmProjects/Digit_classification/model/Digit_classification.model","rb")
# model = pickle.load(pickle_in)
model = tf.keras.models.load_model("C:/Users/Vishwa Raj/PycharmProjects/Digit_classification/model/Digit_classification.model")


# actual testing starts
def preprocess_img(img) :
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.invert(img)
    img = img / 255
    return img

while True :
    success,imgOrg = cap.read()
    if success:
        img = np.asarray(imgOrg)
        img = cv2.resize(img, (28, 28))
        img = preprocess_img(img)
        cv2.imshow("TEST_MODEL", imgOrg)
        img = img.reshape(1, 28, 28)
        print(model.predict_classes(img))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()