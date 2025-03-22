import numpy as np
import sys

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
        self.history = None         # history of the loss function (training and validation loss) for each epoch

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
            x = self._af.activation(np.dot(x, self.weights[i]) + self.biases[i], activation_type=self.activations[i])
        return x

    def __back_propagation(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Perform backpropagation to update weights and biases.

        Parameters:
        x (np.ndarray): Input data.
        y (np.ndarray): True labels.
        """
        nabla_b = [None if b is None else np.zeros(b.shape) for b in self.biases]      # derivative of loss with respect to bias 
        nabla_w = [None if w is None else np.zeros(w.shape) for w in self.weights]     # derivative of loss with respect to weights
        
        # Feed Forward
        a = x
        activations = [x]  # List to store activations for each layer
        zs = [None]            # List to store net-value for each layer
        for i in range(1, len(self.layers)):
            z = np.dot(a, self.weights[i]) + self.biases[i]  # net-value for each layer 
            zs.append(z)
            a = self._af.activation(z, activation_type=self.activations[i])
            activations.append(a)

        # Backward pass
        𝛿 = [np.zeros(a.shape) for a in activations]        # derivative of loss with respect to net
        for i in range(len(self.layers) - 1, 0, -1):        # loop through layers in reverse order excluding input layer
            if i == len(self.layers) - 1:
                𝛿[i] = self._lf.loss_derivative(y, activations[i], loss_type=self.loss_function) * \
                        self._af.activation_derivative(zs[i], activation_type=self.activations[i])
            else:
                𝛿[i] = np.dot(𝛿[i + 1], self.weights[i + 1].T) * \
                        self._af.activation_derivative(zs[i], activation_type=self.activations[i])
                
            nabla_b[i] = 𝛿[i]
            nabla_w[i] = activations[i - 1].reshape(-1, 1) * 𝛿[i]

        return (nabla_b, nabla_w)
            
    def train(self, x: np.ndarray, y: np.ndarray, *, 
              batch_size: int,
              learning_rate: float,
              epochs: int,
              loss_function: str,
              verbose: int = 1,
              validation_split: float = 0.2,
              seed: int = None,
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
        validation_split (float): Fraction of the training data to be used as validation data.
        seed (int): Random seed for shuffling the data.
        """
        self.loss_function = loss_function
        history = np.zeros((epochs, 2))  # Store training and validation loss for each epoch
        num_samples = x.shape[0]
        num_validation_samples = int(num_samples * validation_split)
        num_training_samples = num_samples - num_validation_samples
        x_train = x[:num_training_samples]
        y_train = y[:num_training_samples]
        x_val = x[num_training_samples:]
        y_val = y[num_training_samples:]

        if seed is not None:
            np.random.seed(seed)

        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.arange(num_training_samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            # Train in batches
            for i in range(0, num_training_samples, batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Perform backpropagation for each batch
                nabla_b = [None if b is None else np.zeros(b.shape) for b in self.biases]      # derivative of loss with respect to bias
                nabla_w = [None if w is None else np.zeros(w.shape) for w in self.weights]     # derivative of loss with respect to weights
                for j in range(len(x_batch)):
                    delta_b, delta_w = self.__back_propagation(x_batch[j], y_batch[j])
                    for k in range(1, len(self.layers)):
                        nabla_b[k] += delta_b[k]
                        nabla_w[k] += delta_w[k]

                # Update weights and biases
                for k in range(1, len(self.layers)):
                    self.biases[k] -= (learning_rate / batch_size) * nabla_b[k]
                    self.weights[k] -= (learning_rate / batch_size) * nabla_w[k]

            # Calculate training and validation loss
            train_loss = self._lf.loss(y_train, self.__feed_forward(x_train), loss_type=loss_function)
            val_loss = 0 # self._lf.loss(y_val, self.__feed_forward(x_val), loss_type=loss_function)
            history[epoch] = [train_loss, val_loss]

            if verbose > 0:
                bar_length = 50
                progress = (epoch + 1) / epochs
                bar_filled = "=" * int(progress * bar_length)
                bar_empty = " " * (bar_length - len(bar_filled))
                sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} [{bar_filled}{bar_empty}] "
                                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                sys.stdout.flush()

        self.history = history
        print(f"\nTraining completed. Final Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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
    model.add_layer(2, activation_function='sigmoid', initialization_method='normal', lower_bound=-1, upper_bound=1, seed=25)
    model.add_layer(1, activation_function='sigmoid', initialization_method='normal', mean=0, std=0.1, seed=25)

    # Print the model summary
    # print("Layers:", model.layers)
    # print("Weights:", model.weights)
    # print("Biases:", model.biases)
    # print("Activations:", model.activations)
    # print("Initializations:", model.initialization)
    # print("Weights Gradient:", model.weights_gradient)
    # print("Biases Gradient:", model.biases_gradient)
    # print("Loss Function:", model.loss_function)
    # print("Activation Function:", model._af)
    # print("Loss Function:", model._lf)
    # print("Model:", model)

    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    # y = np.array([[0], [0], [0], [1]])          # AND
    # y = np.array([[1], [1], [1], [0]])          # NAND
    # y = np.array([[0], [1], [1], [1]])          # OR
    # y = np.array([[1], [0], [0], [0]])          # NOR
    # y = np.array([[0], [1], [1], [0]])          # XOR
    y = np.array([[1], [0], [0], [1]])          # XNOR

    print("Input Data:\n", x)
    print("True Labels:\n", y)
    print("Predicted Labels:\n", model.predict(x))

    model.train(x, y, batch_size=1, learning_rate=0.1, epochs=10000, loss_function='MSE', verbose=1, seed=42)
    print("Predicted Labels after training:\n", model.predict(x))
    # print(model.history)

if __name__ == "__main__":
    main()