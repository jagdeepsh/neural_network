import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Literal, Tuple
import math


class CNN:
    def __init__(self, sequence=None):
        self.sequence = sequence

    def forward(self, X):
        return self.sequence.forward(X)

class Sequence:
    def __init__(self, *args):
        self.layers = args

    def forward(self, X):
        self.input = X
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        self.output = A
        return self.output


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
            for j in range(self.output_size[1]):
                # For each RGB in_channel
                for k in range(self.in_channels):
                    # For each row operation for output
                    for l in range(self.output_size[2]):
                        # For each clm operation per row operation
                        for m in range(self.output_size[3]):
                            region = X[i, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            self.output[i, 0, j, l, m] += np.sum(region * self.kernel[j, k])
                            self.output += self.bias(j)
        return self.output

class ReLU:
    def forward(self, X):
        self.values = np.maximum(0, X)
        return self.values
    
    def get_values(self):
        return self.values
    
    def get_derivitives(self, Z):
        return Z > 0


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

        for i in range(self.observations):
            for j in range(self.in_channels):
                for k in range(self.out_channels):
                    for l in range(self.obs_height):
                        for m in range(self.obs_width):
                            range = X[i, j, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            self.output[i, j, k, l, m] += np.max(range)

        return self.output

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

        for i in range(self.observations):
            for j in range(self.in_channels):
                for k in range(self.out_channels):
                    for l in range(self.obs_height):
                        for m in range(self.obs_width):
                            range = X[i, j, k, (l*self.stride):(l*self.stride+self.k1), (m*self.stride):(m*self.stride+self.k2)]
                            self.output[i, j, k, l, m] += np.mean(range)

        return self.output


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
    
    # To revert back to original X input, essentially "get" it back
    def dF(self):
        return self.X

class Categorical_Cross_Entropy_CNN:
    def __init__(self):
        return
    
    def backward(self):
        return

    def cost(self):
        return
    

class Optimizer_CNN:
    def SGD(self, sequence, alpha):
        return
    
    def step(self):
        return
    
