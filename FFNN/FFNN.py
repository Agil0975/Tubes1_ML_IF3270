import numpy as np

import ActivationFunction as af
import LossFunction as lf

class FFNN:
    def __init__(self):
        self.layers = []            # list of integers representing the number of neurons in each layer (including input layer)
        self.weights = []           # list of numpy arrays representing the weights for each layer
        self.biases = []            # list of numpy arrays representing the biases for each layer
        self.activations = []       # list of activation functions for each layer
        self.initialization = []    # initialization method for weights and biases for each layer
        self.weights_gradient = []  # list of numpy arrays representing the gradients of the weights for each layer
        self.biases_gradient = []   # list of numpy arrays representing the gradients of the biases for each layer
        self.loss_function = None   # loss function object

        self._lf = lf.LossFunction()        # Initialize the loss function object
        self._af = af.ActivationFunction()  # Initialize the activation function object

    def add_layer(self, *, 
                  num_neurons, activation_function='sigmoid', 
                  initialization_method='normal', seed=None,
                  lower_bound=0, upper_bound=1,  # for uniform initialization
                  mean=0, std=1,                 # for normal initialization
                 ):
        """
        Add a layer to the neural network.

        Parameters:
            num_neurons (int): Number of neurons in the layer.
            activation_function (str): Activation function for the layer. Default is 'sigmoid'.
            initialization_method (str): Method for initializing weights and biases. Default is 'normal'.
            seed (int): Random seed for initialization. Default is None.
            lower_bound (float): Lower bound for uniform initialization. Default is 0.
            upper_bound (float): Upper bound for uniform initialization. Default is 1.
            mean (float): Mean for normal initialization. Default is 0.
            std (float): Standard deviation for normal initialization. Default is 1.
        """
        self.layers.append(num_neurons)
        self.activations.append(activation_function)
        self.initialization.append(initialization_method)
        if len(self.layers) > 1:
            # Initialize weights and biases for the layer
            self.weights.append(self.__initialize_weights(self.layers[-2], num_neurons, initialization_method,
                                                          seed, lower_bound, upper_bound, mean, std))
            self.biases.append(self.__initialize_biases(num_neurons, initialization_method,
                                                         seed, lower_bound, upper_bound, mean, std))
        else:
            # For the input layer, we don't initialize weights and biases
            self.weights.append(None)
            self.biases.append(None)
    
    def __initialize_weights(self, neurons_in, neurons_out, method, seed=None,
                             lower_bound=0, upper_bound=1, mean=0, std=1):
        """
        Initialize weights for the layer.

        Parameters:
        neurons_in (int): Number of neurons in the previous layer.
        neurons_out (int): Number of neurons in the current layer.
        method (str): Method for initializing weights.

        Returns:
        np.ndarray: Initialized weights.
        """
        if method == 'zero':
            return np.zeros((neurons_in, neurons_out))
        elif method == 'uniform':
            if seed is not None:
                np.random.seed(seed)
            return np.random.uniform(lower_bound, upper_bound, (neurons_in, neurons_out))
        elif method == 'normal':
            if seed is not None:
                np.random.seed(seed)
            return np.random.normal(mean, std, (neurons_in, neurons_out))
        else:
            raise ValueError(f"Unknown initialization method: {method}. Use 'zero', 'uniform', or 'normal'.")

    def __initialize_biases(self, neurons_out, method, seed=None,
                            lower_bound=0, upper_bound=1, mean=0, std=1):
        """
        Initialize biases for the layer.

        Parameters:
        neurons_out (int): Number of neurons in the current layer.
        method (str): Method for initializing biases.

        Returns:
        np.ndarray: Initialized biases.
        """
        if method == 'zero':
            return np.zeros(neurons_out)
        elif method == 'uniform':
            if seed is not None:
                np.random.seed(seed)
            return np.random.uniform(lower_bound, upper_bound, neurons_out)
        elif method == 'normal':
            if seed is not None:
                np.random.seed(seed)
            return np.random.normal(mean, std, neurons_out)
        else:
            raise ValueError(f"Unknown initialization method: {method}. Use 'zero', 'uniform', or 'normal'.")
        
    

# a = 5
# b = 10
# lower_bound = 0
# upper_bound = 10
# mean = 0
# std = 1

# print(f"Uniform initialization: {np.random.uniform(lower_bound, upper_bound, (a, b))}")
# print(f"Normal initialization: {np.random.normal(mean, std, (a, b))}")
# print(f"Zero initialization: {np.zeros((a, b))}")
# print(f"Zero initialization: {np.zeros(a)}")

a = np.array([[1, 2, 3], 
              [4, 5, 6]])
b = np.array([[7, 8, 9],
              [10, 11, 12],
              [13, 14, 15]])
print(np.dot(a, b))  # Transpose b to match the dimensions for dot product