import math
import pandas as pd
import matplotlib.pyplot as plt

def read_data():
    data = pd.read_csv("iris.csv", header=None)
    return data

def target_function(x, theta, bias):
    ans = 0.0
    for i in range (4):
        ans = ans + x[i] * theta[i]
    ans = ans + bias
    return ans

def error_function(prediction, result):
    return (prediction - result) ** 2

def activation_function(pre_prediction):
    return 1/(1+math.exp(-1.0 * pre_prediction))

def theta_gradient(x, theta, bias, result):
    ans = []
    for i in range(4):
        h = activation_function(target_function(x, theta, bias))
        temp = 2 * x[i] * ( result -  h) * (1 - h) * h
        ans.append(temp)
    return ans
    
def bias_gradient(x, theta, bias, result):
    h = activation_function(target_function(x, theta, bias))
    temp = 2 * ( result -  h) * (1 - h) * h
    return temp

def theta_now(theta, theta_grad, learning_rate):
    ans = []
    for i in range(4):
        ans.append(theta[i] + theta_grad[i] * learning_rate)
    return ans
def bias_now(bias, bias_grad, learning_rate):
    return bias + (learning_rate * bias_grad)

data = read_data()
data = data.head(100)
data[4] = data[4].map({'Iris-setosa' : 0.0, 'Iris-versicolor' : 1.0})
theta = [0.5, 0.5, 0.5, 0.5]
bias = 0.5
error = []
print(data.head())
for i in range(60):
    total_error = 0.0
    #er = []
    for index,row in data.iterrows():
        x = row[0:4]
        #print(x)
        #print(row[4])
        function_value = target_function(x, theta, bias)
        #print(function_value)
        after_activated = activation_function(function_value)
        #print(after_activated)
        err = error_function(after_activated, row[4])
        #er.append(err)
        #print(err)
        total_error = total_error + err
        theta_grad = theta_gradient(x, theta, bias, row[4])
        bias_grad = bias_gradient(x, theta, bias, row[4])
        theta = theta_now(theta, theta_grad, 0.8)
        bias = bias_now(bias, bias_grad, 0.8)
    #print(total_error)
    #print(sorted(er))
    print("average error epoch " + str(i+1) + " = " + str(total_error / 100.0) )
    error.append(total_error / 100.0)

plt.plot(error)
plt.ylabel('error')
plt.yscale('log')
plt.show()
