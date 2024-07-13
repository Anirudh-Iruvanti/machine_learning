# Object detection using CNN categorical classification of clothing/accessories images
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D
import matplotlib.pyplot as plt

# Load Features - Image dataset
image_dataset = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = image_dataset.load_data()

# Check data size
print('x_train shape \n', x_train.shape)  # 60000 images of pixel size 28x28
print('x_train 0th image shape \n', x_train[0].shape)  # 1st image of pixel size 28x28

# Show sample image
plt.imshow(x_train[0])  # Image Show: 1st image (shoe)
plt.axis('off')  # Turn off x/y-axis
plt.draw()
plt.show()
plt.imshow(x_train[0], cmap='gray')  # Image Show: 1st image (shoe) - ColourMap = Grayscale
plt.axis('off')  # Turn off x/y-axis
plt.draw()
plt.show()

# Prepare the data

# Feature data preparing
print('x_train[0] matrix values \n\n', x_train[0])  # Print the 28x28 pixel matrix having values: 0 - 255 for the image
print('x_train min', x_train.min())  # Print the min value of the pixel colour
print('x_train max', x_train.max())  # Print the max value of the pixel colour

# 0 < x_train < 255 ; data range varying from 1 digit to 3 digit
# Normalize between 0 and 1 : 0 <= x_train/255 <= 1
x_train = x_train/255
x_test = x_test/255
# As output y are categorical classifiers having 0-9 range, no need to normalize y_train/test data

x_train = x_train.reshape(60000, 28, 28, 1)  # 60000 images of 28x28 convert/reshape to 784x1 1D Array
# For testing we take, say 10000 images
x_test = x_test.reshape(10000, 28, 28, 1)  # 10000 images of 28x28 convert/reshape to 784x1 1D Array

# Target data preparing
print('y_train \n', y_train)  # Array of numbers varying from 0-9
print('y_train size \n', y_train.size)

# Target label names
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Convert 0-9 value to One-hot encoded target classes. eg: One-hot enc of 5 => [0 0 0 0 0 1 0 0 0 0]
y_train_categorical = tf.keras.utils.to_categorical(y_train)

# Check one-hot enc sample values
print('y_train[5]\n', y_train[5], y_train_categorical[5])  # This value is 2
print('y_train_categorical[5] (One-hot encoded)\n', y_train[5], y_train_categorical[5])  # 2nd array value is 1

# Check one-hot enc sample values
print('y_train[3]\n', y_train[3], y_train_categorical[3])  # This value is 3
print('y_train_categorical[3] (One-hot encoded)\n', y_train[3], y_train_categorical[3])  # 3rd array value is 1

# Build NN Layers
model = tf.keras.models.Sequential()

# Convolutional layer
# As we are flattening input x_train (28x28) -> (784,1), here also input_shape we mention the same
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu'))
# Max Pooling -> to reduce loss
model.add(MaxPool2D(pool_size=(2, 2)))
# Flatten
model.add(Flatten())
# Dense Layers
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Output Layer -> We have 10 categories of images => Categorical classifier (softmax)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
              )

print(model.summary())
trained_model = model.fit(x_train, y_train_categorical, epochs=10)

i = 5
predictions = model.predict(x_test[i].reshape(1, 28, 28, 1))
print('Predictions \n', predictions)
predict_max_prob = np.argmax(predictions)
print('Max value in Predictions \n', predict_max_prob)

plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
plt.show()
print('\n The test image is', class_names[predict_max_prob])
