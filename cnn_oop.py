import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Literal, Tuple
import math
from nn_oop import Linear

class CNN:
    def __init__(self, sequence=None):
        self.sequence = sequence

    def forward(self, X):
        return self.sequence.forward(X)

class Sequence:
    def __init__(self, *args):
        self.layers = args

    def forward(self, X):
        self.X = X
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        self.output = A
        return self.output
    
    def get_layers(self):
        return self.layers
    
    def get_X(self):
        return self.X
        


class Conv2D:
    def __init__(self, in_channels=Union[None, int], out_channels=int, kernel_size=Union[int, Tuple[int, int]], stride=Union[None, int], padding=Union[None, Literal['VALID', 'SAME']]):
        if in_channels == None:
            self.in_channels = 1
        else:
            self.in_channels = in_channels

        self.out_channels = out_channels

        if kernel_size == int:
            self.k1 = kernel_size
            self.k2 = kernel_size
            self.kernel_size = (out_channels, in_channels, self.k1, self.k2)
        else:
            self.k1 = kernel_size[0]
            self.k2 = kernel_size[1]
            self.kernel_size = (out_channels, in_channels, self.k1, self.k2)
        
        if stride == None:
            self.stride = 1
        else:
            self.stride = stride
        
        self.padding = padding

        self.kernel = np.random.randn(*self.kernel_size)
        self.bias = np.random.randn(out_channels, 1) #Need to check if dimensions are correct
        return
    
    def modify_input_output(self, X):
        # Setting X for modification for forward propagation
        # X dimension is expanded for consistency of ensuring everything stays 5D
        self.X = X
        if self.padding == 'SAME':

            # Formula for calculating padding requirement for 'SAME' padding
            # Caution that self.kernel_size should be odd number and should be the same
            # Ideally stride should also be 1
            self.p = (self.kernel_size[2] - 1) / 2
            
            # Check number of observations
            if (X.shape.len == 4):
                # Set input dimensions and values
                self.observations = X.shape[0]
                self.obs_height = X.shape[2]
                self.obs_width = X.shape[3]

                # Set output_size
                if self.stride == 1:
                    self.output_size = (self.observations, 1, self.out_channels, self.obs_height, self.obs_width)
                else:
                    self.output_size = (self.observations, 1, self.out_channels, math.floor(((self.obs_height + (2 * self.p) - self.k1) / self.stride) + 1), math.floor(((self.obs_width + (2 * self.p) - self.k2) / self.stride) + 1))

                # For each observation and for each RGB array
                # Set padding for input
                for i, observation in enumerate(range(X.shape[0])):
                    for j, out_channel in enumerate(range(self.out_channels)):
                        X[i, j] =  np.pad(X[i, j], pad_width=self.p, mode='constant', constant_values=0)

            elif(X.shape.len == 3):
                # Set input dimensions and values
                self.obs_height = X.shape[1]
                self.obs_width = X.shape[2]
                if X.shape[0] != 3:
                    # If it is observation by n1 by n2
                    self.observations = X.shape[0]
                    # Set output_size
                    if self.stride == 1:
                        self.output_size = (self.observations, 1, self.out_channels, self.obs_height, self.obs_width)
                    else:
                        self.output_size = (self.observations, 1, self.out_channels, math.floor(((self.obs_height + (2 * self.p) - self.k1) / self.stride) + 1), math.floor(((self.obs_width + (2 * self.p) - self.k2) / self.stride) + 1))
                    # Reshape X for forward propagation
                    X = np.array(self.observations, 1, self.obs_height, self.obs_width)
                    for index, observation in enumerate(range(self.observations)):
                        X[index, 0] = self.X[index]
                    self.X = X
                else:
                    # If it is In_channels by n1 by n2
                    self.observations = 1
                    # Set output_size
                    if self.stride == 1:
                        self.output_size = (1, 1, self.out_channels, self.obs_height, self.obs_width)
                    else:
                        self.output_size = (1, 1, self.out_channels, math.floor(((self.obs_height + (2 * self.p) - self.k1) / self.stride) + 1), math.floor(((self.obs_width + (2 * self.p) - self.k2) / self.stride) + 1))
                    # Modifying X for forward propagation
                    X = np.array(1, 3, self.obs_height, self.obs_width)
                    for index in range(3):
                        X[0, index] = self.X[index]
                    self.X = X

                # Set padding for input
                for i in range(self.X.shape[0]):
                    for j in range(self.X.shape[1]):
                        # Can be either Observation, Height, Width or In_channels, Height, Width
                        X[i, j] = np.pad(X[i, j], pad_width=self.p, mode='constant', constant_values=0)

            elif(X.shape.len == 2):
                # Only 1 observation, and just 1 In_channel
                # Set input dimensions and values
                self.observations = 1
                self.obs_height = X.shape[0]
                self.obs_width = X.shape[1]

                # Set output_size
                if self.stride == 1:
                    self.output_size = (1, 1, self.out_channels, self.obs_height, self.obs_width)
                else:
                    self.output_size = (1, 1, self.out_channels, math.floor(((self.obs_height + (2 * self.p) - self.k1) / self.stride) + 1), math.floor(((self.obs_width + (2 * self.p) - self.k2) / self.stride) + 1))
            
                # Modifying X input for forward propagation
                X = np.array(1, 1, self.obs_height, self.obs_width)
                X[0, 0] = self.X
                self.X = X
                # Set padding for input
                X = np.pad(X[0, 0], pad_width=self.p, mode='constant', constant_values=0)

            else:
                return print('Error, wrong input dimensions')

        elif (self.padding == 'VALID' or self.padding == None):
            # Check number of observations
            if (X.shape.len == 4):
                # Set input dimensions and values
                self.observations = X.shape[0]
                self.obs_height = X.shape[2]
                self.obs_width = X.shape[3]

                # Set output_size
                if self.stride == 1:
                    self.output_size = (self.observations, 1, self.out_channels, (self.obs_height - self.k1 + 1), (self.obs_width - self.k2 + 1))
                else:
                    self.output_size = (self.observations, 1, self.out_channels, math.floor(((self.obs_height - self.k1) / self.stride) + 1), math.floor(((self.obs_width - self.k2) / self.stride) + 1))

                # For each observation and for each RGB array
                # Set padding for input
                for i, observation in enumerate(range(X.shape[0])):
                    for j, out_channel in enumerate(range(self.out_channels)):
                        X[i, j] =  np.pad(X[i, j], pad_width=self.p, mode='constant', constant_values=0)

            elif(X.shape.len == 3):
                # Set input dimensions and values
                self.obs_height = X.shape[1]
                self.obs_width = X.shape[2]
                if X.shape[0] != 3:
                    # If it is observation by n1 by n2
                    self.observations = X.shape[0]
                    # Set output_size
                    if self.stride == 1:
                        self.output_size = (self.observations, 1, self.out_channels, (self.obs_height - self.k1 + 1), (self.obs_width - self.k2 + 1))
                    else:
                        self.output_size = (self.observations, 1, self.out_channel, math.floor(((self.obs_height - self.k1) / self.stride) + 1), math.floor(((self.obs_width - self.k2) / self.stride) + 1))
                    
                    # Modifying X input for forward propagation
                    X = np.array(self.observations, 1, self.obs_height, self.obs_width)
                    for index, observation in enumerate(range(self.observations)):
                        X[index, 0] = self.X[index]
                    self.X = X

                else:
                    # If it is In_channels by n1 by n2
                    self.observations = 1
                    # Set output_size
                    if self.stride == 1:
                        self.output_size = (1, 1, self.out_channels, (self.obs_height - self.k1 + 1), (self.obs_width - self.k2 + 1))
                    else:
                        self.output_size = (1, 1, self.out_channels, math.floor(((self.obs_height - self.k1) / self.stride) + 1), math.floor(((self.obs_width - self.k2) / self.stride) + 1))
                    # Modifying X for forward propagation
                    X = np.array(1, 3, self.obs_height, self.obs_width)
                    for index in range(3):
                        X[0, index] = self.X[index]
                    self.X = X

                # Set padding for input
                for i in range(self.X.shape[0]):
                    for j in range(self.X.shape[1]):
                        # Can be either Observation, Height, Width or In_channels, Height, Width
                        X[i, j] = np.pad(X[i, j], pad_width=self.p, mode='constant', constant_values=0)

            elif(X.shape.len == 2):
                # Only 1 observation, and just 1 In_channel
                # Set input dimensions and values
                self.observations = 1
                self.obs_height = X.shape[0]
                self.obs_width = X.shape[1]

                # Set output_size
                if self.stride == 1:
                    self.output_size = (1, 1, self.out_channels, (self.obs_height - self.k1 + 1), (self.obs_width - self.k2 + 1))
                else:
                    self.output_size = (1, 1, self.out_channels, math.floor(((self.obs_height - self.k1) / self.stride) + 1), math.floor(((self.obs_width - self.k2) / self.stride) + 1))

                # Modifying X for forward propagation
                X = np.array(1, 1, self.obs_height, self.obs_width)
                X[0, 0] = self.X
                self.X = X

                # Set padding for input
                X = np.pad(X[0, 0], pad_width=self.p, mode='constant', constant_values=0)

            else:
                return print('Error, wrong input dimensions')

        else:
            return print('Error with padding method')

        self.output = np.empty(*self.output_size)
        return
        
    def forward(self, X):
        # Reshape X for padding and creating output_size
        self.modify_input_output(X)

        # Forward propagation
        # For each observation
        for i in range(self.output_size[0]):
            # For each kernel
            for j in range(self.out_channels):
                # For each RGB in_channel
                for k in range(self.in_channels):
                    # For each row operation for output
                    for l in range(self.output_size[3]):
                        # For each clm operation per row operation
                        for m in range(self.output_size[4]):
                            region = X[i, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            self.output[i, 0, j, l, m] += np.sum(region * self.kernel[j, k])
                            self.output += self.bias(j)
        return self.output

    def get_kernel(self):
        return self.kernel
    
    def get_bias(self):
        return self.bias
    
    def find_and_set_dK(self, dZ):
        self.dZ = dZ
        # input dimensions are: (obs, in, none, n1, n2) or (obs, 1, out, n1, n2)
        # dZ dimensions are: (obs, 1, out, n1, n2)
        # dK should be: case 1: (obs, 1, out, n1, n2), case 2: (obs, 1, out, n1, n2)
        self.dK_observations = self.dZ.shape[0]
        self.dK_in_channels = self.dZ.shape[1]
        self.dK_out_channels = self.dZ.shape[2]
        self.dK_height = (self.X.shape[3] - self.dZ.shape[3]) + 1
        self.dK_width = (self.X.shape[4] - self.dZ.shape[4]) + 1
        self.dK_size = (self.dK_observations, self.dK_in_channels, self.dK_out_channels, self.dK_height, self.dK_width)
        self.dK = np.empty(*self.dK_size)

        dZ_K1 = dZ.shape[3]
        dZ_K2 = dZ.shape[4]

        # Case 1: (Input)
        if (len(self.X.shape) != 5):
            for i in range(self.dK_observations):
                for j in range(self.dK_out_channels):
                    for k in range(self.dK_in_channels):
                        for l in range(self.dK_height):
                            for m in range(self.dK_width):
                                region = self.X[i, k, (l):(l+dZ_K1), (m):(m+dZ_K2)]
                                self.dK[i, 0, j, l, m] += np.sum(region * dZ[i, j, k])
        
        # Case 2 (From previous Activation / Pooling Function)
        else:
            for i in range(self.dK_observations):
                for j in range(self.dK_out_channels):
                    for k in range(self.dK_height):
                        for l in range(self.dK_width):
                            region = self.X[i, 0, j, (k):(k+dZ_K1), (l):(l+dZ_K2)]
                            self.dK[i, 0, j, l, m] += np.sum(region * dZ[i, j, k])
        
        return

    def find_and_set_dI(self):
        padding = ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1))
        dZ_padded = self.dZ
        dZ_padded = np.pad(dZ_padded, pad_width=padding, mode='constant', constant_values=0)
        self.dI_observations = self.dZ.shape[0]
        self.dI_in_channels = self.dZ.shape[1]
        self.dI_out_channels = self.dZ.shape[2]
        self.dI_height = self.dZ.shape[3] + 2 - self.k1 + 1
        self.dI_width = self.dZ.shape[4] + 2 - self.k2 + 1
        self.dI_size = (self.dI_observations, self.dI_in_channels, self.dI_out_channels, self.dI_height, self.dI_width)
        self.dI = np.empty(*self.dI_size)

        kernel_rotated = self.kernel
        kernel_rotated = np.rot90(kernel_rotated, k=2)

        for i in range(self.dI_observations):
            for j in range(self.dI_out_channels):
                for k in range(self.dI_height):
                    for l in range(self.dI_width):
                        region = dZ_padded[i, 0, j, (k):(k+self.k1), (l):(l+self.k2)]
                        self.dI[i, 0, j, k, l] += np.sum(region * kernel_rotated[j, 0])
        return

    def get_dK(self):
        return self.dK
    
    def get_dB(self):
        return self.dB

    def set_dB(self, dB):
        self.dB = dB
        return

    def set_dZ(self, dZ):
        self.dZ = dZ
        return
    
    def get_dZ(self):
        return self.dZ

    def SGD_step_kernel_bias(self, alpha):
        self.kernel = self.kernel - (alpha * self.dK)
        self.bias = self.bias - (alpha * self.dB)
        return

class ReLU:
    def forward(self, X):
        self.values = np.maximum(0, X)
        return self.values
    
    def get_values(self):
        return self.values
    
    def get_derivitives(self, Z):
        return Z > 0

class Softmax:
    def forward(self, X):
        X = X - np.max(X, axis=1, keepdims=True)
        np_exp = np.exp(X)
        self.values = np_exp / np.sum(np_exp, axis=1, keepdims=True)
        return self.values
    
    def get_values(self):
        return self.values

class MaxPool2D:
    def __init__(self, kernel_size=Union[int, Tuple[int, int]], stride=Union[None, int]):
        self.kernel_size = kernel_size
        if self.kernel_size == int:
            self.k1 = self.kernel_size
            self.k2 = self.kernel_size
        else:
            self.k1 = self.kernel_size[0]
            self.k2 = self.kernel_size[1]
        
        if stride != kernel_size:
            self.stride = stride
        else:
            self.stride = kernel_size
        return

    def forward(self, X):
        # Setting observation dimensions
        self.X = X
        self.observations = X.shape[0]
        self.in_channels = X.shape[1]
        self.out_channels = X.shape[2]
        self.obs_height = X.shape[3]
        self.obs_width = X.shape[4]

        # Setting output_size and array
        self.output_height = (((self.obs_height - self.k1) / self.stride) + 1)
        self.output_width = (((self.obs_width - self.k2) / self.stride) + 1)
        self.output_size = (self.observations, self.in_channels, self.out_channels, self.output_height, self.output_width)
        self.output = np.empty(*self.output_size)

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                for k in range(self.output_size[2]):
                    for l in range(self.output_size[3]):
                        for m in range(self.output_size[4]):
                            range = X[i, j, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            self.output[i, j, k, l, m] += np.max(range)

        return self.output

    def find_and_set_dP(self, dZ):
        self.dP = np.zeros(*self.X.shape)
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                for k in range(self.output_size[2]):
                    for l in range(self.output_size[3]):
                        for m in range(self.output_size[4]):
                            range = self.X[i, j, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            max_position = np.argmax(range)
                            self.dP[max_position] += dZ[i, j, k, l, m]
        return self.dP

    def set_dP(self, dP):
        self.dP = dP
        return
    
    def get_dP(self):
        return self.dP

class AvePool2D:
    def __init__(self, kernel_size=Union[int, Tuple[int, int]], stride=Union[None, int]):
        self.kernel_size = kernel_size
        if self.kernel_size == int:
            self.k1 = self.kernel_size
            self.k2 = self.kernel_size
        else:
            self.k1 = self.kernel_size[0]
            self.k2 = self.kernel_size[1]
        
        if stride != kernel_size:
            self.stride = stride
        else:
            self.stride = kernel_size
        return

    def forward(self, X):
        # Setting observation dimensions
        self.X = X
        self.observations = X.shape[0]
        self.in_channels = X.shape[1]
        self.out_channels = X.shape[2]
        self.obs_height = X.shape[3]
        self.obs_width = X.shape[4]

        # Setting output_size and array
        self.output_height = (((self.obs_height - self.k1) / self.stride) + 1)
        self.output_width = (((self.obs_width - self.k2) / self.stride) + 1)
        self.output_size = (self.observations, self.in_channels, self.out_channels, self.output_height, self.output_width)
        self.output = np.empty(*self.output_size)

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                for k in range(self.output_size[2]):
                    for l in range(self.output_size[3]):
                        for m in range(self.output_size[4]):
                            range = X[i, j, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            self.output[i, j, k, l, m] += np.mean(range)

        return self.output

    def find_and_set_dP(self, dZ):
        self.dP = np.zeros(*self.X.shape)
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                for k in range(self.output_size[2]):
                    for l in range(self.output_size[3]):
                        for m in range(self.output_size[4]):
                            ave_amount = dZ[i, j, k, l, m] / (self.output_size[3] * self.output_size[4])
                            range = self.dP[i, j, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            range[:] = ave_amount
        return self.dP
    
    def get_dP(self):
        return self.dP
    
class flatten:

    def forward(self, X):
        # Setting X dimension values
        self.X = X
        self.observations = X.shape[0]
        self.in_channels = X.shape[1]
        self.out_channels = X.shape[2]
        self.obs_height = X.shape[3]
        self.obs_width = X.shape[4]

        # n represents the number of total neurons per observation for the input layer in the fully connected layer
        self.n = (self.out_channels * self.obs_height * self.obs_width)
        X_reshaped = X.reshape(self.observations, self.out_channels, self.obs_height, self.obs_width)
        self.output = X_reshaped.reshape(self.observations, self.n)
        self.output = self.output.T

        return self.output
    
    def get_dZ(self):
        return self.dZ
    
    def set_dZ(self, dZ):
        self.dZ = dZ
        return
    
    def get_shape(self):
        return self.X.shape

class Categorical_Cross_Entropy_CNN:
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

            # Fully Connected Layers Backpropagation onwards:::
            if isinstance(layer, Linear):
                # Last Fully Connected layer (Output side)
                if index == 1:
                    A = self.layers[index+1].get_values()

                    dZi = self.output - self.labels
                    dWi = (1/m) * dZi.dot(A.T)
                    dBi = (1/m) * np.sum(dZi, axis=1, keepdims=True)

                    layer.set_dZ(dZi)
                    layer.set_dW(dWi)
                    layer.set_dB(dBi)

                # First Fully Connected layer
                elif isinstance(self.layers[index+1], flatten):
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
                    

                # Middile Fully Connected Layers
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


            # CNN Backpropagation onwards:::
            # derivitive of flatten method
            elif(isinstance(layer, flatten)):
                dZ = self.layers[index-1].get_weight().T * self.layers[index-1].get_dZ()
                dZ = dZ.reshape(layer.get_shape())
                layer.set_dZ(dZ)

            # If current index is a Pooling layer
            elif(isinstance(layer, MaxPool2D) or isinstance(layer, AvePool2D)):
                dZ = self.layers[index-1].get_dZ()
                dP = layer.find_and_set_dP(dZ)

            # Setting the derivitive of ReLU activation layer
            elif(isinstance(layer, Conv2D)):
                if isinstance(self.layers[index-2], MaxPool2D) or isinstance(self.labels[index-2], AvePool2D):
                    dP = self.layer[index-2].get_dP()
                else:
                    dP = self.layer[index-1].get_dZ()
                dF = self.layers[index-1].get_derivitives()
                dZ = dP.dot(dF)

                dB = np.sum(dZ)

                layer.find_and_set_dK(dZ)
                layer.set_dZ(dZ)
                layer.set_dB(dB)

                # If it is not first Convolutional Layer, find dI
                if index != (len(self.layers) - 1):
                    layer.find_and_set_dI()


    # Currently Labels is designed for One_Hot_Y
    # One_Hot_Y to be n by m
    def cost(self):
        output_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        predicted_values = np.sum(output_clipped*self.labels, axis=0)
        self.cost_amount = np.mean(-np.log(predicted_values))
        return self.cost_amount
    
class One_Hot_Y:
    # Y should be 1 by m
    # Objective is to turn Y into n by m
    def __init__(self, Y):
        # Convert Y into n by m of zeros
        One_Hot_Values = np.zeros(Y.max(), Y.size)
        # Converts values to 1 at given rows and clmns
        One_Hot_Values[Y, np.arange(Y.size)] = 1
    
    def get_one_hot_Y(self):
        return self.Y


class Optimizer_CNN:
    def SGD(self, sequence, alpha):
        self.sequence = sequence
        self.alpha = alpha
    
    def step(self):
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer.SGD_step_kernel_bias(self.alpha)
            if isinstance(layer, Linear):
                layer.SGD_step_weights_bias(self.alpha)
        return
    
