import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(10)

# Working with data:::
def get_data(path):
    data = pd.read_csv(path)
    return data

def convert_datafram_to_array(data):
    data = np.array(data)
    m, n = data.shape
    return m, n, data

def data_split(data):
    np.random.shuffle(data)

    data_dev = data[2941:3361].T #10%
    data_train = data[0:2940].T #70%
    data_test = data[3362:].T #10%

    X_dev = data_dev[1:]
    X_train = data_train[1:]
    X_test = data_test[1:]

    Y_dev = data_dev[0]
    Y_train = data_train[0]
    Y_test = data_test[0]

    return X_dev, X_test, X_train, Y_dev, Y_test, Y_train



# Creating model:::
# Layer creation, forward propation, activation functions
def layer(n_input, n_neurons):
    weight = np.random.randn(n_neurons, n_input)
    bias = np.random.randn(n_neurons, 1)
    return weight, bias

def forward(input, weight, bias, activation):
    z = weight.dot(input) + bias
    output = activation(z)
    return output

def ReLU(input):
    return np.maximum(0, input)

def softmax(input):
    np_exp = np.exp(input - np.max(input, axis=1, keepdims=True))
    return np_exp / np.sum(np_exp, axis=1, keepdims=True)

# Optimising model:::
# Cost, One_Hot_Y, accuracy, prediction, Backprop, differentiation, StochasticGradientDescend
def cost(input, One_Hot_Y):
    # Make sure One_Hot_Y is n by m
    input = np.clip(input, 1e-15, 1 - 1e-15)
    correct_confidence = np.sum(input * One_Hot_Y, axis=1)
    return np.mean(-np.log(correct_confidence))

def backprop(weights, biases, activation_funciton, A, X, One_Hot_Y):
    # Weights, biases and activatiosn should be in descending order from output to input
    # Y Should be One_Hot_Y
    # Return type would be a list of arrays for the dW and dB respectively
    dWs = []
    dBs = []
    dZs = []
    m, n = One_Hot_Y.shape
    for i in len(A):
        # First derivitive for Softmax Function
        if (i == 0):
            dZi = A[i] - One_Hot_Y
            dWi = 1/m * dZi.dot(A[i+1].T)
            dBi = 1/m * np.sum(dZi, axis=1, keepdims=True)
        # Subsequent derivitives for layers and ReLU activations
        elif (i != len(A)):
            dZi = weights[i-1].dot(dZs[i-1]) * derive_ReLU(dZs[i-1])
            dWi = 1/m * dZi.dot(A[i+1].T)
            dBi = 1/m * np.sum(dZi, axis=1, keepdims=True)
        # Final derivitive for first layer with input X
        else:
            dZi = weights[i-1].dot(dZs[i-1]) * derive_ReLU(dZs[i-1])
            dWi = 1/m * dZi.dot(X.T)
            dBi = 1/m * np.sum(dZi, axis=1, keepdims=True)
            dWs.append(dWi)
            dBs.append(dBs)
            return dWs, dBs
        dWs.append(dWi)
        dBs.append(dBs)
        dZs.append(dZi)


def one_hot(Y):
    # Creates 2D array of 0's, m by n
    One_Hot_Y = np.zeroes((Y.size, Y.max() + 1))
    One_Hot_Y = One_Hot_Y[np.arange(Y.size), Y] = 1
    return One_Hot_Y.T

def derive_ReLU(input):
    return input > 0

def accuracy(predictions, Y):
    return np.sum(prediciton == Y) / Y.size

def prediciton(A):
    return np.argmax(A, axis=1)

def StochasticGradientDescend(weights, biases, dWs, dBs, alpha):
    for i in len(weights):
        weights[i] = weights[i] - alpha * dWs[i]
        biases[i] = biases[i] - alpha * dBs[i]
    return weights, biases

# Saving model:::
def save_model(weights, biases):
    for i in len(weights):
        df = pd.DataFrame(weights[i])
        df.to_csv(f'W{len(weights)-i}.csv')
        df = pd.DataFrame(biases[i])
        df.to_csv(f'B{len(biases)-i}.csv')


# Load model:::
def load_model(index):
    weights = []
    biases = []
    for i in index:
        weight = pd.read_csv(f'W{i+1}.csv')
        bias = pd.read_csv(f'B{i+1}.csv')
        weight = np.array(weight)
        bias = np.array(bias)
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def plot_model(costs, accuracy):
    plt.plot(costs)
    plt.plot(accuracy)
    return

# Train model:::
def train_model(index, iterations, overwrite, X, Y):
    weights, biases = load_model(index)
    
    for i in iterations:
        pass

    return
# Params -> Path to load, no. of iterations, should read, train, read results, predict, accuracy call, plot, ask to save
# Auto save param perhaps, can have yes or no input section


if __name__ == '__main__':

    # Working with data
    data = get_data('train.csv')
    m, n, data = convert_datafram_to_array(data)
    X_dev, X_test, X_train, Y_dev, Y_test, Y_train = data_split(data)

    # Creating model
    W1, B1 = layer(784, 10)
    A1 = forward(X_train, W1, B1, ReLU)
    W2, B2 = layer(10, 10)
    forward(A1, W2, B2, softmax)

    # Optimising model


    # Saving model


    # Loading model


    # Training model