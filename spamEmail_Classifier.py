from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from tabulate import tabulate
from scipy.special import expit


# read files
D_tr = genfromtxt('spambasetrain.csv', delimiter = ',', encoding = 'utf-8')
D_ts = genfromtxt('spambasetest.csv', delimiter = ',', encoding = 'utf-8')

# construct x and y for training and testing
X_tr = D_tr[: ,: -1]
y_tr = D_tr[: , -1]
X_ts = D_ts[: ,: -1]
y_ts = D_ts[: , -1]

# number of training / testing samples
n_tr = D_tr.shape[0]
x = D_tr.shape[1]
n_ts = D_ts.shape[0]
y = D_ts.shape[1]

X_tr = np.concatenate((np.ones((n_tr, 1)), X_tr), axis = 1)
X_ts = np.concatenate((np.ones((n_ts, 1)), X_ts), axis = 1)

# lr = 1e-3


def predict(X, w):
    n_ts = X.shape[0]
    # use w for prediction
    pred = np.zeros(n_ts)       # initialize prediction vector
    sigmoid1 = np.zeros(n_ts)
    '''for i in range(n_ts):
        for j in range(X.shape[1]):
            temp[i] = temp[i] + w[j] * X[i][j]'''
    s1 = np.array(w)
    temp1 = np.matmul(X, s1)

    for j in range(n_ts):
        sigmoid1[j] = 1 / (1 + np.exp(temp1[j]))

    for i in range(n_ts):
        prob = sigmoid1[i]
        if prob > 0.5:
            pred[i] = 0             # compute your prediction
        else:
            pred[i] = 1
    return pred


def accuracy(X, y, w):
    y_pred = predict(X, w)
    total_correct_predictions = 0
    total_predictions = y_pred.shape[0]
    for p in range(y_pred.shape[0]):
        if y_pred[p] == y[p]:
            total_correct_predictions = total_correct_predictions + 1
    data_set_accuracy = (total_correct_predictions/total_predictions)*100
    return data_set_accuracy


def logistic_reg(X_tr, X_ts, lr, y_tr, y_ts, reg_model, reg_coeff):

    # perform gradient ascent

    n_vars = X_tr.shape[1]  # number of variables
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.01        # tolerance for stopping

    iter = 0                # iteration counter
    max_iter = 1000       # maximum iteration
    train_accuracy_iterations = []
    testing_accuracy_iterations = []

    while True:
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars)  # initialize gradient
        sigmoid = np.zeros(X_tr.shape[0])

        '''for i in range(X_tr.shape[0]):
            for j in range(X_tr.shape[1]):
                temp[i] = temp[i] + (w[j] * X_tr[i][j])'''

        w_array = np.array(w)
        temp = np.matmul(X_tr, w_array)

        if lr == 1:
            if iter > 1:
                break
            else:
                for l in range(X_tr.shape[0]):
                    sigmoid[l] = np.exp(temp[l]) / (1 + np.exp(temp[l]))

        else:
            for l in range(X_tr.shape[0]):
                sigmoid[l] = np.exp(temp[l]) / (1 + np.exp(temp[l]))

        if reg_model == "Regularized":
            reg = reg_coeff * w_array

        equation1 = np.matmul(y_tr, X_tr)
        equation2 = np.matmul(sigmoid, X_tr)

        for j in range(n_vars):
            if reg_model == "nonRegularized":
                grad[j] = equation1[j] - (equation2[j])
            elif reg_model == "Regularized":
                grad[j] = equation1[j] - (equation2[j]) - (reg[j])

        w_new = w + (lr * grad)

        # stopping criteria and perform update if not stopping
        if lr == 1:
            if iter <= 1:
                print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))
                train_accuracy = accuracy(X_tr, y_tr, w)
                test_accuracy = accuracy(X_ts, y_ts, w)
                train_accuracy_iterations.append([iter, train_accuracy])
                testing_accuracy_iterations.append([iter, test_accuracy])
                w = w_new

        if np.mean(np.abs(grad)) < tolerance:
            train_accuracy = accuracy(X_tr, y_tr, w)
            test_accuracy = accuracy(X_ts, y_ts, w)
            train_accuracy_iterations.append([iter, train_accuracy])
            testing_accuracy_iterations.append([iter, test_accuracy])
            w = w_new
            break
        else:
            w = w_new

        if iter % 50 == 0:
            print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))
            train_accuracy = accuracy(X_tr, y_tr, w)
            test_accuracy = accuracy(X_ts, y_ts, w)
            train_accuracy_iterations.append([iter, train_accuracy])
            testing_accuracy_iterations.append([iter, test_accuracy])

        if iter >= max_iter:
            break

    return train_accuracy_iterations, testing_accuracy_iterations


learning_rates = np.array([10**0, 10**-2, 10**-4, 10**-6])
train_accuracy_results = {}
testing_accuracy_results = {}

for lr in range(len(learning_rates)):
    train_accuracy, test_accuracy = logistic_reg(X_tr, X_ts, learning_rates[lr], y_tr, y_ts, "nonRegularized", 0)
    train_accuracy_results.update({learning_rates[lr]: train_accuracy})
    testing_accuracy_results.update({learning_rates[lr]: test_accuracy})

print("train accuracy results")
print(train_accuracy_results)
print("test accuracy results")
print(testing_accuracy_results)


coeff = np.array([-8, -6, -4, -2, 0, 2], dtype=float)
reg_coeff = []
train_accuracy_results_regularized = {}
testing_accuracy_results_regularized = {}

for i in range(len(coeff)):
    reg_coeff.append(2**(coeff[i]))

for rc in range(len(reg_coeff)):
    train_accuracy_regularized, test_accuracy_regularized = logistic_reg(X_tr, X_ts, 10**-4, y_tr, y_ts, "Regularized", reg_coeff[rc])
    train_accuracy_results_regularized.update({reg_coeff[rc]: train_accuracy_regularized})
    testing_accuracy_results_regularized.update({reg_coeff[rc]: test_accuracy_regularized})

print("training accuracy after regularization")
print(train_accuracy_results_regularized)
print("testing accuracy after regularization")
print(testing_accuracy_results_regularized)

for i in train_accuracy_results_regularized[2**(-8)]:
    if i[0] == 1000:
        train_val_reg = i[1]

for i in testing_accuracy_results_regularized[2**(-8)]:
    if i[0] == 1000:
        test_val_reg = i[1]

for m in train_accuracy_results[10**-4]:
    if m[0] == 1000:
        train_val_non_reg = m[1]

for m in testing_accuracy_results[10**-4]:
    if m[0] == 1000:
        test_val_non_reg = m[1]

print ("Training and Testing Accuracy for regularized and non regularized LR models with best learning rate of 10**-4 and regularized coefficient of -8 for 1000th iteration")

print(tabulate([['Regularized LR', train_val_reg, test_val_reg], ['Non-regularized LR', train_val_non_reg, test_val_non_reg]], headers=['logistic regression model', 'Training Accuracy', 'Testing Accuracy']))

# plots for all learning rates
x1 = []
y1 = []
x2 = []
y2 = []

for k in train_accuracy_results:
    for l in train_accuracy_results[k]:
        x1.append(l[0])
        y1.append(l[1])
        # plotting the line 1 points
    plt.xticks(x1)
    plt.plot(x1, y1, label="Training Accuracy")

    # line 2 points
    for m in testing_accuracy_results[k]:
        x2.append(m[0])
        y2.append(m[1])
        # plotting the line 2 points
    plt.xticks(x2)
    plt.plot(x2, y2, label="Testing Accuracy")

    # naming the x axis
    plt.xlabel('No of iterations')
    # naming the y axis
    plt.ylabel('Accuracy')
    # giving a title to my graph
    plt.title('Learning rate : {0}'.format(str(k)))

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()
    x1 = []
    y1 = []
    x2 = []
    y2 = []


# plot for logistic regression with regularization
x3 = []
y3 = []
x4 = []
y4 = []

for rc in train_accuracy_results_regularized:
    t = train_accuracy_results_regularized[rc]
    last_train = t[-1]
    x3.append(np.log2(rc))
    y3.append(last_train[1])

    h = testing_accuracy_results_regularized[rc]
    last_test = h[-1]
    x4.append(np.log2(rc))
    y4.append(last_test[1])

plt.xticks(x3)
plt.plot(x3, y3, label="Training Accuracy")
plt.xticks(x4)
plt.plot(x4, y4, label="Testing Accuracy")
plt.xlabel('Regularization Coefficients')
# naming the y axis
plt.ylabel('Accuracy')
# giving a title to my graph
plt.title('Logistic regression Regularization')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()

