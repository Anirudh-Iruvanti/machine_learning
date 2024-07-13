# XOR Gate Logic Model built as Neural Network
# No hidden layers

import numpy as np
import tensorflow as tf

# For known inputs, and known outputs lets create features, targets
# 2 input XOP gate all possible input combinations
features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Corresponding XOR gate outputs
targets = np.array([[0], [1], [1], [0]])

# Build the NN
# Define the NN model as sequential
model = tf.keras.models.Sequential()
# Input layer, (here we are not considering hidden layer)
model.add(tf.keras.layers.Dense(4, activation='relu'))
# Output layer. No.of outputs =1 (0 or 1), so sigmoid
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile -> To add the loss function, optimization algo, metrics
model.compile(loss=tf.keras.losses.MeanSquaredError,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
              )

# Link the training input to training output, Create fit() to use
# input, output and fit the curve/model. Define the epoch (iter)
trained_model = model.fit(features, targets, epochs=1000)
print('\n ---- Model training completed ---- \n')

# Predictions
test_inputs = np.array([[0, 0], [0, 1]])
print('Test input data \n', test_inputs)
predictions = model.predict(test_inputs)
predict_round = predictions.round()
print('Predicted output data \n', predictions)
print('Predicted output data (rounded) \n', predict_round)
