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

    def add_layer(self, num_neurons, *,
                  activation_function='sigmoid', 
                  initialization_method='normal', 
                  seed=None,
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
            self.initialization[-1] = None  # No initialization for the input layer
            self.activations[-1] = None     # No activation function for the input layer
        
        # placeholder for gradients
        self.weights_gradient.append(None)
        self.biases_gradient.append(None)
    
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
        
    def __feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform feedforward operation.

        Parameters:
        x (np.ndarray): Input data.

        Returns:
        np.ndarray: Output of the neural network.
        """
        for i in range(1, len(self.layers)):
            x = self._af.activation(x @ self.weights[i] + self.biases[i],
                                  activation_function=self.activations[i])
        return x

    def __back_propagation(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Perform backpropagation to update weights and biases.

        Parameters:
        x (np.ndarray): Input data.
        y (np.ndarray): True labels.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]      # derivative of loss with respect to bias 
        nabla_w = [np.zeros(w.shape) for w in self.weights]     # derivative of loss with respect to weights
        
        # Feed Forward
        a = x
        activations = [x]  # List to store activations for each layer
        zs = []            # List to store net-value for each layer
        for i in range(1, len(self.layers)):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)
            a = self._af.activation(z, activation_function=self.activations[i])
            activations.append(a)

        # Backward pass
        𝛿 = [np.zeros(a.shape) for a in activations]        # derivative of loss with respect to net
        for i in range(len(self.layers) - 1, 0, -1):        # loop through layers in reverse order excluding input layer
            if i == len(self.layers) - 1:
                𝛿[i] = self._lf.lost_derivative(y, activations[i], loss_function=self.loss_function) * \
                        self._af.activation_derivative(zs[i], activation_function=self.activations[i])
            else:
                𝛿[i] = (𝛿[i + 1] @ self.weights[i + 1].T) * \
                        self._af.activation_derivative(zs[i], activation_function=self.activations[i])
                
            nabla_b[i] = 𝛿[i]
            nabla_w[i] = activations[i - 1] * 𝛿[i].reshape(-1, 1)

        return (nabla_b, nabla_w)
            
    def train(self, x: np.ndarray, y: np.ndarray, *, 
              batch_size: int,
              learning_rate: float,
              epochs: int,
              loss_function: str,
              verbose: int = 1,
              validation_split: float = 0.2,
              ):
        """
        Train the neural network.
        
        Parameters:
        x (np.ndarray): Input data.
        y (np.ndarray): True labels.
        batch_size (int): Size of each batch for training.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs for training.
        loss_function (str): Loss function to be used for training.
        verbose (int): Verbosity mode. 0 = silent, 1 = progress bar + training info

        returns:
        np.ndarray: history of the loss function (training and validation loss) for each epoch
        """
        self.loss_function = loss_function
        
        

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters:
        x (np.ndarray): Input data.

        Returns:
        np.ndarray: Predicted labels.
        """
        return self.__feed_forward(x)
    
    def plot_network_graph(self):
        """
        Plot the network graph.
        """
        pass

    def plot_weight_distribution(layers: list[int]):
        """
        Plotting the weight distribution for a specific layer

        Parameters:
        layers (list[int]): List of layers to be plotted
        """
        pass

    def plot_gradient_distribution(layers: list[int]):
        """
        Plotting the gradient weight distribution for a specific layer

        Parameters:
        layers (list[int]): List of layers to be plotted
        """
        pass

    def save(self, filename: str):
        """
        Save the model network to a file.

        Parameters:
        filename (str): Name of the file to save the neural network.
        """
        pass

    def load(self, filename: str):
        """
        Load the model network from a file.

        Parameters:
        filename (str): Name of the file to load the neural network.
        """
        pass

def main():
    """
    Main function to test the FFNN class.
    """
    # Create an instance of the FFNN class
    model = FFNN()

    # Add layers to the model
    model.add_layer(2)
    model.add_layer(2, activation_function='sigmoid', initialization_method='uniform', lower_bound=-1, upper_bound=1)
    model.add_layer(1, activation_function='relu', initialization_method='normal', mean=0, std=0.1)

    # Print the model summary
    print("Layers:", model.layers)
    print("Weights:", model.weights)
    print("Biases:", model.biases)
    print("Activations:", model.activations)
    print("Initializations:", model.initialization)
    print("Weights Gradient:", model.weights_gradient)
    print("Biases Gradient:", model.biases_gradient)
    print("Loss Function:", model.loss_function)
    print("Activation Function:", model._af)
    print("Loss Function:", model._lf)
    print("Model:", model)

    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    # y = np.array([[0], [0], [0], [1]])          # AND
    y = np.array([[1], [1], [1], [0]])        # NAND
    # y = np.array([[0], [1], [1], [1]])        # OR
    # y = np.array([[1], [0], [0], [0]])        # NOR
    # y = np.array([[0], [1], [1], [0]])        # XOR
    # y = np.array([[1], [0], [0], [1]])        # XNOR

    print("Input Data:\n", x)
    print("True Labels:\n", y)
    print("Predicted Labels:\n", model.predict(x))
    𝛿 = 10
    print(𝛿)

    arr1 = np.array([1, 10, 100])
    arr2 = np.array([1, -1]).reshape(-1, 1)
    print(arr1)
    print(arr2)
    print(arr1 * arr2)

if __name__ == "__main__":
    main()

    import sys
    import time

    # def progress_bar(total, length=200):
    #     for i in range(total + 1):
    #         progress = i / total
    #         bar = '=' * int(progress * length)
    #         spaces = ' ' * (length - len(bar))
    #         sys.stdout.write(f"\r[{bar}{spaces}] {int(progress * 100)}%")
    #         sys.stdout.flush()
    #         time.sleep(0.1)
    #     print()

    # progress_bar(100)
