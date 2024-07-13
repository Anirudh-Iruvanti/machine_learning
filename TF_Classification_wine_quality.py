# Classification of Beverage/Wine Quality (Good/Bad) using Tensorflow

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Prepare the data

df = pd.read_csv('data/winequalityN.csv')  # datafile df
sample_dat = df.sample(5)
print('sample data 5 rows \n', sample_dat)  # sample data of 5 rows
sample_shape = df.shape
print('csv data size \n', sample_shape)  # csv data size row x column
missing_values = df.isnull().sum()  # checking csv for sum of missing values in each column
print('Missing values sum each column \n', missing_values)
# As no.of missing data is insignificant, drop the rows from df
df = df.dropna()  # drop NaN (Null data)
missing_values = df.isnull().sum()  # checking csv for missing values dropped or not
print('Missing values sum each column after redundant data dropped \n', missing_values)
# From dataset, for "type" column, check data value count
type_val_cnt = df['type'].value_counts()
print('Value count "Type" \n', type_val_cnt)
df['is_white_wine'] = [1 if typ == 'white' else 0 for typ in df['type']]
print(df.sample(5))
# As we have used type column and generated is_white_wine, we don't need it anymore
# axis = 1 => column operation, inplace to replace change in df
df.drop('type', axis=1, inplace=True)
print(df.head())
qual_val_cnt = df['quality'].value_counts()
print('Value count "quality" \n', qual_val_cnt)
# Threshold of quality = 6; >= 6 is good quality
df['is_good_wine'] = [1 if qual >= 6 else 0 for qual in df['quality']]
print(df.sample(5))
df.drop('quality', axis=1, inplace=True)
print(df.head())

# For Train/Test - Split the data

# Our target variable is "is_Good_wine". Our features will be everything besides target variable
# Dropping the target var from var X (not modifying df)
X = df.drop('is_good_wine', axis=1)
# Target variable kept in y
y = df['is_good_wine']

# test_size is 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Feature data size \n", X_train.shape, X_test.shape)

# Pre-processing
# Sulphates, citric acid are near to 0, total Sulphur dioxide is 3 digits. Data re-scaling TBD
scaler = StandardScaler()
# Apply fit and transform to X_train
X_train_scaled = scaler.fit_transform(X_train)
# Let's only transform X_test
X_test_scaled = scaler.transform(X_test)

# check for 3 rows
print(X_train_scaled[:3])

# Model training
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train_scaled, y_train, epochs=100)
print('\n ---- Model training finished ---- \n')
# ----- Model Training Finished -----------

# Model evaluation
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

plt.plot(np.arange(1, 101), history.history['loss'], label='Loss')
plt.plot(np.arange(1, 101), history.history['accuracy'], label='Accuracy')
plt.plot(np.arange(1, 101), history.history['precision'], label='Precision')
plt.plot(np.arange(1, 101), history.history['recall'], label='Recall')
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()

# Predictions
predictions = model.predict(X_test_scaled)
print(predictions)

prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
print(prediction_classes[:20])

print('Confusion Matrix \n [ TP FP  \n   FN TN ]\n')
print(confusion_matrix(y_test, prediction_classes))

# Metrics/Scores
print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Precision: {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Recall: {accuracy_score(y_test, prediction_classes):.2f}')
