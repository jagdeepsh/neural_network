import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class NeuralNetwork:
    def __init__(self, sequence=None):
        self.sequence = sequence
        self.layers = sequence.get_layers()
        return
    
    def forward(self, X):
        self.outputs = self.sequence.forward(X)
        return self.outputs
    
    def outputs(self):
        return self.outputs

    # Might have to restructure "params"
    def save_as_csv(self, path=None):
        for index, layer in enumerate(self.layers):
            if (isinstance(layer, Linear)):
                weight = layer.get_weight()
                bias = layer.get_bias()
                data_weight = pd.DataFrame(weight)
                data_weight.to_csv(f'W{index}.csv')
                data_bias = pd.DataFrame(bias)
                data_bias.to_csv(f'B{index}.csv')

    # Current design and implementation is for Multi Class Classification
    def predict(self, X):
        A = self.sequence.forward(X)
        return np.argmax(A, axis=0)
    
    def accuracy(self, prediction, Y):
        # Y passed here should not be One_Hot_Y rather just the 1 by m
        return np.mean(prediction == Y)

    def plot(self, cost, accuracy):
        cost_iteration = cost.size
        accuracy_iteration = accuracy.size
        x_axis = np.arange(accuracy_iteration)
        plt.plot(x_axis, cost, label='Cost')
        plt.plot(x_axis, accuracy, label='Accuracy')
        plt.title("Ave. Cost & Ave. Accuracy Analysis")
        plt.xlabel("Iteration")
        plt.ylabel("Ave. Cost / Ave. Accuracy")
        plt.legend()
        plt.show()



class Linear:
    def __init__(self, n_input=None, n_output=None):
        self.n_input = n_input
        self.n_output = n_output
        self.W = np.random.randn(n_output, n_input)
        self.B = np.random.randn(n_output, 1)

    def get_weight(self):
        return self.W
    
    def get_bias(self):
        return self.B
    
    def set_weight(self, W):
        self.W = W

    def set_bias(self, B):
        self.B = B
    
    def forward(self, X):
        self.Z = self.W.dot(X) + self.B
        return self.Z
    
    def get_Z(self):
        return self.Z
    
    def set_dW(self, dW):
        self.dW = dW
    
    def get_dW(self):
        return self.dW
    
    def set_dB(self, dB):
        self.dB = dB
    
    def get_dB(self):
        return self.dB
    
    def set_dZ(self, dZ):
        self.dZ = dZ
    
    def get_dZ(self):
        return self.dZ
    
    def SGD_step_weights_bias(self, alpha):
        self.W = self.W - (alpha * self.dW)
        self.B = self.B - (alpha * self.dB)


class Sequence:
    def __init__(self, *args):
        self.layers = args

    def get_params(self):
        # W and B are arrays of W1, B1, W2, B2...,
        # where each row represents each layer
        # A represents the activation values for backpropagation
        # Z represents the Z value generated at each layer
        # Top bottom is the first to last activation and layer
        W = []
        B = []
        A = []
        Z = []

        for index, layer in enumerate(self.layers):
            if (isinstance(layer, Linear)):
                W.append(layer.get_weight())
                B.append(layer.get_bias())
                Z.append(layer.get_Z())
            elif(isinstance(layer, ReLU) or isinstance(layer, Softmax)):
                A.append(layer.get_values())

        return W, B, A, Z
    
    def get_layers(self):
        return self.layers

    def forward(self, X):
        self.X_input = X
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def get_X(self):
        return self.X_input
    

class Classification_Cross_Entropy:
    def __init__(self, output=None, labels=None):
        self.output = output
        self.labels = labels

    def backward(self, sequence):
        self.sequence = sequence
        self.layers = self.sequence.get_layers()
        # Layers are now output to input
        self.layers = self.layers[::-1]
        X = sequence.get_X()
        n, m = self.labels.shape

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                # Last layer (Output side)
                if index == 1:
                    A = self.layers[index+1].get_values()

                    dZi = self.output - self.labels
                    dWi = (1/m) * dZi.dot(A.T)
                    dBi = (1/m) * np.sum(dZi, axis=1, keepdims=True)

                    layer.set_dZ(dZi)
                    layer.set_dW(dWi)
                    layer.set_dB(dBi)

                # First layer (Intput side)
                elif index == len(self.layers) - 1:
                    previous_layer_weight = self.layers[index-2].get_weight()
                    previous_layer_dZ = self.layers[index-2].get_dZ()
                    Z = layer.get_Z()
                    activation_derivitive = self.layers[index - 1].get_derivitive(Z)

                    dZi = previous_layer_weight.T.dot(previous_layer_dZ) * activation_derivitive
                    dWi = (1/m) * dZi.dot(X.T)
                    dBi = (1/m) * np.sum(dZi, axis=1, keepdims=True)

                    layer.set_dZ(dZi)
                    layer.set_dW(dWi)
                    layer.set_dB(dBi)

                # Middile layers
                else:
                    previous_layer_weight = self.layers[index-2].get_weight()
                    previous_layer_dZ = self.layers[index-2].get_dZ()
                    Z = layer.get_Z()
                    activation_derivitive = self.layers[index - 1].get_derivitive(Z)
                    A = self.layers[index+1].get_values()

                    dZi = previous_layer_weight.T.dot(previous_layer_dZ) * activation_derivitive
                    dWi = (1/m) * dZi.dot(A.T)
                    dBi = (1/m) * np.sum(dZi, axis=1, keepdims=True)

                    layer.set_dZ(dZi)
                    layer.set_dW(dWi)
                    layer.set_dB(dBi)


    # Currently Labels is designed for One_Hot_Y
    # One_Hot_Y to be n by m
    def cost(self):
        output_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        predicted_values = np.sum(output_clipped*self.labels, axis=1)
        self.cost_amount = np.mean(-np.log(predicted_values))
        return self.cost_amount
    

class Mean_Squared_Error:
    def __init__(self, output=None, labels=None):
        # Labels here should be 1 by m
        self.output = output
        self.labels = labels
        self.n, self.m = self.labels.shape

    def backward(self, sequence):
        self.sequence = sequence
        self.layers = self.sequence.get_layers()
        # Layers are now output to input
        self.layers = self.layers[::-1]
        X = sequence.get_X()
        n, m = self.labels.shape

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                # Last layer (Output side)
                if index == 1:
                    A = self.layers[index+1].get_values()

                    dZi = -2(self.labels - layer.get_Z())
                    dWi = (1/m) * dZi.dot(A.T)
                    dBi = (1/m) * np.sum(dZi, axis=1, keepdims=True)

                    layer.set_dZ(dZi)
                    layer.set_dW(dWi)
                    layer.set_dB(dBi)

                # First layer (Input side)
                elif index == len(self.layers) - 1:
                    previous_layer_weight = self.layers[index-2].get_weight()
                    previous_layer_dZ = self.layers[index-2].get_dZ()
                    Z = layer.get_Z()
                    activation_derivitive = self.layers[index - 1].get_derivitive(Z)

                    dZi = previous_layer_weight.T.dot(previous_layer_dZ) * activation_derivitive
                    dWi = (1/m) * dZi.dot(X.T)
                    dBi = (1/m) * np.sum(dZi, axis=1, keepdims=True)

                    layer.set_dZ(dZi)
                    layer.set_dW(dWi)
                    layer.set_dB(dBi)

                # Middile layers
                else:
                    previous_layer_weight = self.layers[index-2].get_weight()
                    previous_layer_dZ = self.layers[index-2].get_dZ()
                    Z = layer.get_Z()
                    activation_derivitive = self.layers[index - 1].get_derivitive(Z)
                    A = self.layers[index+1].get_values()

                    dZi = previous_layer_weight.T.dot(previous_layer_dZ) * activation_derivitive
                    dWi = (1/m) * dZi.dot(A.T)
                    dBi = (1/m) * np.sum(dZi, axis=1, keepdims=True)

                    layer.set_dZ(dZi)
                    layer.set_dW(dWi)
                    layer.set_dB(dBi)


    # Currently Labels is designed for One_Hot_Y
    # One_Hot_Y to be n by m
    def cost(self):
        self.cost_amount = (1/self.m) * np.sum(np.square(self.labels - self.output))
        return self.cost_amount


class Optimizer:
    def SGD(self, sequence, alpha):
        self.sequence = sequence
        self.alpha = alpha
        self.layers = sequence.get_layers()
    
    def step(self):
        for layer in self.layers:
            if (isinstance(layer, Linear)):
                layer.SGD_step_weights_bias(self.alpha)
        return




class One_Hot_Y:
    # As of now this is designed for multi class classification
    # Where the Y is 1 by m
    # Objective is to create One_Hot_Y that is n by m
    def __init__(self, Y):
        One_Hot_Y_Values = np.zeros((Y.max() + 1, Y.size))
        One_Hot_Y_Values[Y, np.arange(Y.size)] = 1
        self.Y = One_Hot_Y_Values
    def get_one_hot_Y(self):
        return self.Y

class ReLU:
    def forward(self, X):
        self.values = np.maximum(0, X)
        return self.values
    
    def get_values(self):
        return self.values
    
    def get_derivitive(self, Z):
        return Z > 0
    
class Sigmoid:
    def forward(self, X):
        X = np.clip(X, 1e-7, 1 - 1e-7)
        self.values = 1 / (1 + np.exp(-X))
        return self.values
    
    def get_values(self):
        return self.values
        
    def get_derivitive(self, Z):
        Z = np.clip(Z, 1e-7, 1 - 1e-7)
        return -(1 + np.exp(-Z)) * (-np.exp(-Z))
    

class Softmax:
    def forward(self, X):
        X = X - np.max(X, axis=1, keepdims=True)
        np_exp = np.exp(X)
        self.values = np_exp / np.sum(np_exp, axis=1, keepdims=True)
        return self.values
    
    def get_values(self):
        return self.values
    


if __name__ == '__main__':
    # Set Data and Create model here
    # Create layers and activation functions
    # Create a sequence by passing layers and activation funcitons within it
    # Pass sequence to NeuralNetwork and call forward
    # Create a One_Hot_Y which would be your labels
    # Create a cost object by passing your output from the forward call and the labels
    # Calculate the cost, call backpropagte
    # Instantiate a Optimise object, for now use SGD = Stochastic Gradient Descend
    # Call step on optimise by passing in your cost object


    # Data management
    iris = load_iris()
    data = iris.data # m by n
    Y = iris.target # 1 by m
    m, n = data.shape # 150, 4

    data_train = data[0:113].T
    X_train = data_train
    Y_train = Y[0:113]

    data_test = data[114:].T
    X_test = data_test
    Y_test = Y[114:]


    # Creating layers and Sequence
    layer1 = Linear(4, 10)
    sigmoid1 = Sigmoid()
    layer2 = Linear(10, 3)
    softmax1 = Softmax()

    sequence = Sequence(layer1, sigmoid1, layer2, softmax1)

    # Creating labels
    labels = One_Hot_Y(Y_train).get_one_hot_Y()

    # Creating model
    model = NeuralNetwork(sequence)
    
    

    # Training / testing / plotting / saving model
    no_iterations = 100
    cost_amounts = np.zeros(no_iterations)
    accuracy_amounts = np.zeros(no_iterations)
    for i in range(no_iterations):
        output = model.forward(X_train)
        ave_cost = Classification_Cross_Entropy(output, labels)
        ave_cost.backward(sequence)
        optimise = Optimizer()
        optimise.SGD(sequence, 0.04)
        optimise.step()
        
        prediction = model.predict(X_train)
        ave_accuracy_amount = model.accuracy(prediction, Y_train)
        ave_cost_amount = ave_cost.cost()
        cost_amounts[i] = ave_cost_amount
        accuracy_amounts[i] = ave_accuracy_amount

        if (i % 2 == 0):
            print(f'Epoch: {i},\nCost: {ave_cost_amount},\nAccuracy: {ave_accuracy_amount}')
        
        if (i == no_iterations - 1):
            model.save_as_csv('test.csv')

    # To plot cost and accuracy graph
    model.plot(cost_amounts, accuracy_amounts)