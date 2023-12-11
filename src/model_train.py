import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
##########################################
# X_train -> contains all the hand written digits in form of array ( 28,28 )
# Y_train -> label for what number it is for the corresponding X_train
# X_test & Y_test same as above just lesser amount of samples to test the model
##########################################

# loading pre processed data set from MNIST
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

#scaling everything to normalize the training data to 0->1
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

 #building up the NN model
 model = tf.keras.models.Sequential()
 model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
 model.add(tf.keras.layers.Dense(128, activation='relu'))
 model.add(tf.keras.layers.Dense(100, activation='relu'))
 model.add(tf.keras.layers.Dense(10, activation='softmax'))

 model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

 #train the model with the data
 model.fit(X_train, Y_train, epochs=10)

##save the model to use it
#model.save("C:/Users/Vishwa Raj/PycharmProjects/Digit_classification/model/Digit_classification.model")

##evaluation of the model
#loss, accuracy = model.evaluate(X_test, Y_test)
#print(loss)
#print(accuracy)

#test
#model = tf.keras.models.load_model("C:/Users/Vishwa Raj/PycharmProjects/Digit_classification/model/Digit_classification.model")
#X_flat = X_test[5000].reshape(1,28,28)
#y = model.predict(X_flat)
#plt.matshow(X_test[5000])
#plt.show()
#print("Y_TEST:",Y_test[5000])
#print("predict:",np.argmax(y))




