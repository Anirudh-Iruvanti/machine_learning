# Linear regression for 2 features to predict student grade using Gradient Descent (Batch)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_dat = pd.read_csv('student_grades.csv')
print('csv data \n', csv_dat)

print('Null \n', csv_dat.isnull())
print('Number of missing values \n', csv_dat.isnull().sum())

csv_dat = csv_dat.fillna(csv_dat.median())  # Fill null with median
print('updated csv data \n', csv_dat)

x1_y = plt.scatter(csv_dat.study_hours_daily, csv_dat.grades_out_of_10)
plt.title("Feature x1 vs actual target y")
plt.xlabel("study_hours_daily")
plt.ylabel("grades_out_of_10")
plt.show()
x2_y = plt.scatter(csv_dat.class_attendance, csv_dat.grades_out_of_10)
plt.title("Feature x2 vs actual target y")
plt.xlabel("class_attendance")
plt.ylabel("grades_out_of_10")
plt.show()

df = np.array(csv_dat)  # csv to array, df = dataframe

print('csv to array \n', df)
print('data shape \n', df.shape)
print('data size \n', df.size)

m = df.shape[0]  # no.of rows in data set
print('dataset size', m)

x1_orig = df[0:m, 0]  # feature 1
x2_orig = df[0:m, 1]  # feature 2

x1_mean = np.mean(x1_orig)
x2_mean = np.mean(x2_orig)

x1 = (x1_orig - x1_mean)/(np.max(x1_orig) - np.min(x1_orig))  # Mean normalized x1
x2 = (x2_orig - x2_mean)/(np.max(x2_orig) - np.min(x2_orig))  # Mean normalized x2

y = df[0:m, 2]  # actual target values
print('Feature x1 (study hours daily) \n', x1)
print('Feature x2 (class attendance) \n', x2)
print('Targets y (grades out of 10) \n', y)

# Model function f(x) = w1*x1 + w2*x2 + b
# Cost function J(w,b) = (1/2m)*[summation (f(xi) - yi) ** 2] ; (i = 1 to m)


def gradient_descent_algo(w_1, w_2, x_1, x_2, bias, y_actual, m_num, learn_rate):

    dj_dw1 = 0
    dj_dw2 = 0
    dj_db = 0
    #  Summation from i = 0 to m-1
    for i in range(0, m_num):
        dj_dw1 += (1/m_num)*(((w_1*x_1[i]) + bias) - y_actual[i])*x_1[i]
        dj_dw2 += (1/m_num)*(((w_2*x_2[i]) + bias) - y_actual[i])*x_2[i]
        dj_db += (1/m_num)*(((w_1*x_1[i]) + bias) - y_actual[i])
    # print('sum of partial derivative J(w1,b) wrt w1', dj_dw1)
    # print('sum of partial derivative J(21,b) wrt w2', dj_dw2)
    # print('sum of partial derivative J(w1,b) wrt b', dj_db)
    w1_new = w_1 - learn_rate * dj_dw1
    w2_new = w_2 - learn_rate * dj_dw2
    b_new = bias - learn_rate * dj_db
    return w1_new, w2_new, b_new


w1 = 0
w2 = 0
b = 0
lr = 0.05
epoch = 50

for e in range(0, epoch):
    w1, w2, b = gradient_descent_algo(w1, w2, x1, x2, b, y, m, lr)
    print(f'w1: {w1}, w2: {w2}, b: {b}, epoch: {e}')

print(f'Optimal model parameters are w1: {w1},  w2: {w2}, b: {b}')


def trained_model(w1_optimized, w2_optimized, b_optimized, x1_test_value,  x2_test_value):
    f_predict = w1_optimized*x1_test_value + w2_optimized*x2_test_value + b_optimized  # Trained model
    return f_predict


k = 5  # test value
x1_test = x1[k]
x2_test = x2[k]
y_test = trained_model(w1, w2, b, x1_test, x2_test)
print(f'\n Predicted value is yhat {y_test} and actual y value is {y[k]}')
