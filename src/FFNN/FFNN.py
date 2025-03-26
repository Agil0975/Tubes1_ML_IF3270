import numpy as np
import sys
import matplotlib.pyplot as plt
import networkx as nx
import pickle

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
        self.rmsNorm = False        # do RMS or not

        self._lf = lf.LossFunction()        # Initialize the loss function object
        self._af = af.ActivationFunction()  # Initialize the activation function object

    def add_layer(self, num_neurons, *,
                  activation_function='sigmoid', 
                  activation_parameters={},
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
            activation_parameters (dict): Parameters for the activation function. Default is an empty dictionary.
            initialization_method (str): Method for initializing weights and biases. Default is 'normal'.
            seed (int): Random seed for initialization. Default is None.
            lower_bound (float): Lower bound for uniform initialization. Default is 0.
            upper_bound (float): Upper bound for uniform initialization. Default is 1.
            mean (float): Mean for normal initialization. Default is 0.
            std (float): Standard deviation for normal initialization. Default is 1.
        """
        self.layers.append(num_neurons)
        self.activations.append((activation_function, activation_parameters))
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
        if seed is not None:
            np.random.seed(seed)

        if method == 'zero':
            return np.zeros((neurons_in, neurons_out))
        
        elif method == 'uniform':
            return np.random.uniform(lower_bound, upper_bound, (neurons_in, neurons_out))
        
        elif method == 'normal':
            return np.random.normal(mean, std, (neurons_in, neurons_out))
        
        elif method == 'xavier_uniform':
            limit = np.sqrt(6 / (neurons_in + neurons_out))
            return np.random.uniform(-limit, limit, (neurons_in, neurons_out))
        
        elif method == 'xavier_normal':
            std = np.sqrt(2 / (neurons_in + neurons_out))
            return np.random.normal(0, std, (neurons_in, neurons_out))
        
        elif method == 'he_uniform':
            limit = np.sqrt(3 / neurons_in)
            return np.random.uniform(-limit, limit, (neurons_in, neurons_out))
        
        elif method == 'he_normal':
            std = np.sqrt(2 / neurons_in)
            return np.random.normal(0, std, (neurons_in, neurons_out))
        
        else:
            raise ValueError(f"Unknown initialization method: {method}. Use 'zero', 'uniform', 'normal', "
                             "'xavier_uniform', 'xavier_normal', 'he_uniform', or 'he_normal'.")

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
        if seed is not None:
            np.random.seed(seed)

        if method in ['zero', 'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', None]:
            return np.zeros(neurons_out)

        elif method == 'uniform':            
            return np.random.uniform(lower_bound, upper_bound, neurons_out)
        
        elif method == 'normal':
            return np.random.normal(mean, std, neurons_out)

        else:
            raise ValueError(f"Unknown initialization method: {method}. Use 'zero', 'uniform', 'normal', "
                             "'xavier_uniform', 'xavier_normal', 'he_uniform', or 'he_normal'.")
    
    def __rms_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Perform RMS normalization on the input data.

        Parameters:
        x (np.ndarray): Input data.

        Returns:
        np.ndarray: Normalized data.
        """
        rms = np.sqrt(np.mean(x ** 2, axis=0, keepdims=True)) + 1e-8
        return x / rms

    def __feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform feedforward operation.

        Parameters:
        x (np.ndarray): Input data.

        Returns:
        np.ndarray: Output of the neural network.
        """
        for i in range(1, len(self.layers)):
            z = np.dot(x, self.weights[i]) + self.biases[i]  # net-value for each layer
            if self.rmsNorm:
                z = self.__rms_norm(z)
            x = self._af.activation(z, activation_type=self.activations[i][0], **self.activations[i][1])
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
        zs = [None]        # List to store net-value for each layer
        rms_zs = [None]    # List to store RMS normalized net-value for each layer
        for i in range(1, len(self.layers)):
            z = np.dot(a, self.weights[i]) + self.biases[i]  # net-value for each layer
            zs.append(z)

            if self.rmsNorm:
                rms_z = self.__rms_norm(z)
                rms_zs.append(rms_z)
                a = self._af.activation(rms_z, activation_type=self.activations[i][0], **self.activations[i][1])  # activation function for each layer
            else:
                a = self._af.activation(z, activation_type=self.activations[i][0], **self.activations[i][1])   # activation function for each layer

            activations.append(a)

        # Backward pass
        𝛿 = [np.zeros(a.shape) for a in activations]        # derivative of loss with respect to net
        for i in range(len(self.layers) - 1, 0, -1):        # loop through layers in reverse order excluding input layer
            if i == len(self.layers) - 1:
                if self.rmsNorm:
                    pass
                else:
                    𝛿[i] = self._lf.loss_derivative(y, activations[i], loss_type=self.loss_function) * \
                            self._af.activation_derivative(zs[i], activation_type=self.activations[i][0], **self.activations[i][1])
            else:
                if self.rmsNorm:
                    pass
                else:
                    𝛿[i] = np.dot(𝛿[i + 1], self.weights[i + 1].T) * \
                            self._af.activation_derivative(zs[i], activation_type=self.activations[i][0], **self.activations[i][1])
                
            nabla_b[i] = 𝛿[i]
            nabla_w[i] = activations[i - 1].reshape(-1, 1) * 𝛿[i]

        return (nabla_b, nabla_w)
            
    def train(self, 
              x_train: np.ndarray, y_train: np.ndarray, 
              x_val: np.ndarray = None, y_val: np.ndarray = None, 
              *, 
              batch_size: int,
              learning_rate: float,
              epochs: int,
              loss_function: str,
              l1_lambda: float = 0,
              l2_lambda: float = 0,
              verbose: int = 1,
              seed: int = None,
              error_threshold: float = 0,
              rmsNorm: bool = False,
              ):
        """
        Train the neural network.
        
        Parameters:
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation data. If None, validation will be done using the training data.
        y_val (np.ndarray): Validation labels. If None, validation will be done using the training labels.
        batch_size (int): Size of each batch for training.
        learning_rate (float): Learning rate for the optimizer.
        l1_lambda (float): L1 regularization parameter.
        l2_lambda (float): L2 regularization parameter.
        epochs (int): Number of epochs for training.
        loss_function (str): Loss function to be used for training.
        verbose (int): Verbosity mode. 0 = silent, 1 = progress bar + training info
        seed (int): Random seed for shuffling the data.
        error_threshold (float): Threshold for early stopping based on validation loss.
        rmsNorm (bool): Whether to use RMS normalization or not.
        """
        self.loss_function = loss_function
        self.rmsNorm = rmsNorm
        history = np.zeros((epochs, 2))  # Store training and validation loss for each epoch
        num_training_samples = x_train.shape[0]

        if x_val is None or y_val is None:
            x_val = x_train
            y_val = y_train

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

                # Update weights and biases with regularization
                for k in range(1, len(self.layers)):
                    # L1 regularization
                    if l1_lambda != 0:
                        nabla_w[k] += l1_lambda * np.sign(self.weights[k])

                    # L2 regularization
                    if l2_lambda != 0:
                        nabla_w[k] += 2 * l2_lambda * self.weights[k]

                    # Update weights and biases
                    self.biases[k] -= (learning_rate / batch_size) * nabla_b[k]
                    self.weights[k] -= (learning_rate / batch_size) * nabla_w[k]

            # Calculate training and validation loss
            train_loss = self._lf.loss(y_train, self.__feed_forward(x_train), loss_type=loss_function)
            val_loss = self._lf.loss(y_val, self.__feed_forward(x_val), loss_type=loss_function)
            history[epoch] = [train_loss, val_loss]

            if verbose > 0:
                bar_length = 50
                progress = (epoch + 1) / epochs
                bar_filled = "=" * int(progress * bar_length)
                bar_empty = " " * (bar_length - len(bar_filled))
                sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} [{bar_filled}{bar_empty}] {progress:.2%}, "
                                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                sys.stdout.flush()

            if val_loss < error_threshold:
                print(f"\nStopping early at epoch {epoch + 1} due to validation loss below threshold.")
                break
                
        self.weights_gradient = nabla_w
        self.biases_gradient = nabla_b
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

    def plot_network_graph(self, *, visualize="both"):
        """
        Plot the network graph, showing network structure with numeric weight, gradient, bias, and bias gradient labels.
        Parameters:
        visualize (str): Type of visualization. 
            Options:
            - weights: Show network weights
            - gradients: Show network weight gradients
            - both: Show both weights and gradients (default)
        """

        if visualize not in ["weights", "gradients", "both"]:
            raise ValueError('Invalid visualization type. Use "weights", "gradients", or "both".')

        # Create a directed graph
        G = nx.DiGraph()
        
        # Track node positions for consistent layout
        pos = {}
        layer_heights = {}

        # Create nodes for each layer
        node_labels = {}
        for layer_idx, neurons in enumerate(self.layers):
            # Calculate vertical spacing for neurons in this layer
            layer_heights[layer_idx] = np.linspace(0, 1, neurons + 2)[1:-1]
            
            for neuron_idx in range(neurons):
                node_name = f"L{layer_idx}_N{neuron_idx}"
                
                # Position nodes horizontally by layer, vertically by neuron count
                pos[node_name] = (layer_idx, layer_heights[layer_idx][neuron_idx])
                
                # Add node to graph
                G.add_node(node_name)
                
                # Create node labels with bias information
                if layer_idx > 0:  # Skip input layer
                    bias = self.biases[layer_idx][neuron_idx] if self.biases[layer_idx] is not None else 0
                    bias_gradient = self.biases_gradient[layer_idx][neuron_idx] if self.biases_gradient[layer_idx] is not None else 0
                    
                    # Construct label based on visualization type
                    if visualize == "weights":
                        # Single type visualization
                        node_labels[node_name] = f"B: {bias:.4f}"
                    elif visualize == "gradients":
                        # Single type visualization
                        node_labels[node_name] = f"BG: {bias_gradient:.4f}"
                    elif visualize == "both":
                        # Both weights and gradients
                        node_labels[node_name] = f"B: {bias:.4f}\nBG: {bias_gradient:.4f}"
                else:
                    # Input layer nodes remain empty
                    node_labels[node_name] = ""
        
        # Add edges between layers with weights
        edge_labels = {}
        for layer_idx in range(1, len(self.layers)):
            for src_neuron in range(self.layers[layer_idx-1]):
                for dest_neuron in range(self.layers[layer_idx]):
                    src_node = f"L{layer_idx-1}_N{src_neuron}"
                    dest_node = f"L{layer_idx}_N{dest_neuron}"
                    
                    # Determine edge labels based on visualization type
                    if visualize in ["weights", "both"]:
                        weight = self.weights[layer_idx][src_neuron, dest_neuron]
                        edge_labels[(src_node, dest_node)] = f"W: {weight:.4f}"
                    
                    if visualize in ["gradients", "both"]:
                        gradient = self.weights_gradient[layer_idx][src_neuron, dest_neuron] if self.weights_gradient[layer_idx] is not None else 0
                        
                        # If showing both, append gradient to existing label
                        if visualize == "both":
                            edge_labels[(src_node, dest_node)] += f"\nWG: {gradient:.4f}"
                        else:
                            edge_labels[(src_node, dest_node)] = f"WG: {gradient:.4f}"
                    
                    # Add edge to graph
                    G.add_edge(src_node, dest_node)
        
        # Create the plot
        plt.figure(figsize=(12, 6.75))
        plt.title(f"Neural Network Graph - {visualize.capitalize()} Visualization")
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=1500)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10)
        
        # Draw node labels (bias and bias gradient information)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5)
        
        # Draw edge labels (weight and weight gradient information)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, label_pos=0.75)
        
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_weight_distribution(self, layers=None):
        """
        Plotting the weight and bias distribution for specified layers.

        Parameters:
        layers (list[int], optional): List of layer indices to plot. 
                                      If None, plots all layers except input layer.
        """

        # If no layers specified, default to all layers except input layer
        if layers is None:
            layers = range(1, len(self.layers))
        
        # Filter out layers that are out of range or have None weights/biases
        valid_layers = [
            layer for layer in layers 
            if layer < len(self.layers) and 
            self.weights[layer] is not None and 
            self.biases[layer] is not None
        ]

        # If no valid layers, raise an informative error
        if not valid_layers:
            raise ValueError("No valid layers found to plot. Ensure layer indices are correct and weights/biases are initialized.")
        
        # Calculate number of rows needed (2 layers per row)
        num_layers = len(valid_layers)
        num_rows = (num_layers + 1) // 2

        # Create figure with 4 columns per row (2 for weights, 2 for biases)
        fig, axs = plt.subplots(num_rows, 4, figsize=(12, 3*num_rows))
        
        # Adjust subplot spacing for better readability
        plt.subplots_adjust(
            top=0.95,
            bottom=0.05,
            left=0.05,
            right=0.95,
            hspace=0.3,
            wspace=0.3
        )
        
        # Add main title to the figure
        fig.suptitle("Weight and Bias Distribution", 
                    fontsize=16, fontweight="bold")
        
        # Ensure axs is a 2D array for consistent indexing
        if num_rows == 1:
            axs = axs.reshape(1, -1)
        
        # Iterate through rows (each row contains 2 layers)
        for row in range(num_rows):
            # Determine which layers to plot in this row
            layer_indices = valid_layers[row*2:row*2+2]
            
            # Iterate through layers in this row
            for col, layer_index in enumerate(layer_indices):
                # Flatten weights and extract biases
                weights = self.weights[layer_index].flatten()
                biases = self.biases[layer_index]
                
                # Plot weight distribution histogram
                axs[row, col*2].hist(weights, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
                axs[row, col*2].set_title(f"Layer {layer_index} - Weight Distribution", fontsize=10)
                axs[row, col*2].set_xlabel("Weight", fontsize=8)
                axs[row, col*2].set_ylabel("Frequency", fontsize=8)
                axs[row, col*2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                axs[row, col*2].tick_params(axis="both", which="major", labelsize=7)
                
                # Calculate and display weight distribution statistics
                weight_mean = np.mean(weights)
                weight_std = np.std(weights)
                axs[row, col*2].text(0.95, 0.95, 
                                    f"Mean: {weight_mean:.4f}\nStd: {weight_std:.4f}", 
                                    transform=axs[row, col*2].transAxes, 
                                    verticalalignment="top", 
                                    horizontalalignment="right",
                                    fontsize=7,
                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
                
                # Plot bias distribution histogram
                axs[row, col*2+1].hist(biases, bins=50, edgecolor="black", alpha=0.7, color="lightgreen")
                axs[row, col*2+1].set_title(f"Layer {layer_index} - Bias Distribution", fontsize=10)
                axs[row, col*2+1].set_xlabel("Bias", fontsize=8)
                axs[row, col*2+1].set_ylabel("Frequency", fontsize=8)
                axs[row, col*2+1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                axs[row, col*2+1].tick_params(axis="both", which="major", labelsize=7)
                
                # Calculate and display bias distribution statistics
                bias_mean = np.mean(biases)
                bias_std = np.std(biases)
                axs[row, col*2+1].text(0.95, 0.95, 
                                    f"Mean: {bias_mean:.4f}\nStd: {bias_std:.4f}", 
                                    transform=axs[row, col*2+1].transAxes, 
                                    verticalalignment="top", 
                                    horizontalalignment="right",
                                    fontsize=7,
                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        
        # Hide any unused subplots
        for row in range(num_rows):
            for col in range(4):
                if not axs[row, col].has_data():
                    axs[row, col].axis("off")
        
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(layers: list[int]):
        """
        Plotting the gradient weight distribution for a specific layer

        Parameters:
        layers (list[int]): List of layers to be plotted
        """
        pass

    def plot_gradient_distribution(self, layers=None):
        """
        Plotting the weight and bias gradient distribution for specified layers.

        Parameters:
        layers (list[int], optional): List of layer indices to plot. 
                                      If None, plots all layers except input layer.
        """

        # If no layers specified, default to all layers except input layer
        if layers is None:
            layers = range(1, len(self.layers))
        
        # Filter out layers that are out of range or have None weight/bias gradients
        valid_layers = [
            layer for layer in layers 
            if layer < len(self.layers) and 
            self.weights_gradient[layer] is not None and 
            self.biases_gradient[layer] is not None
        ]

        # If no valid layers, raise an informative error
        if not valid_layers:
            raise ValueError("No valid layers found to plot. Ensure layer indices are correct and gradients are computed.")
        
        # Calculate number of rows needed (2 layers per row)
        num_layers = len(valid_layers)
        num_rows = (num_layers + 1) // 2

        # Create figure with 4 columns per row (2 for weight gradients, 2 for bias gradients)
        fig, axs = plt.subplots(num_rows, 4, figsize=(12, 3*num_rows))
        
        # Adjust subplot spacing for better readability
        plt.subplots_adjust(
            top=0.95,
            bottom=0.05,
            left=0.05,
            right=0.95,
            hspace=0.3,
            wspace=0.3
        )
        
        # Add main title to the figure
        fig.suptitle("Weight and Bias Gradient Distribution", 
                    fontsize=16, fontweight="bold")
        
        # Ensure axs is a 2D array for consistent indexing
        if num_rows == 1:
            axs = axs.reshape(1, -1)
        
        # Iterate through rows (each row contains 2 layers)
        for row in range(num_rows):
            # Determine which layers to plot in this row
            layer_indices = valid_layers[row*2:row*2+2]
            
            # Iterate through layers in this row
            for col, layer_index in enumerate(layer_indices):
                # Flatten weight gradients and extract bias gradients
                weight_gradients = self.weights_gradient[layer_index].flatten()
                bias_gradients = self.biases_gradient[layer_index]
                
                # Plot weight gradient distribution histogram
                axs[row, col*2].hist(weight_gradients, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
                axs[row, col*2].set_title(f"Layer {layer_index} - Weight Gradient Distribution", fontsize=10)
                axs[row, col*2].set_xlabel("Weight Gradient", fontsize=8)
                axs[row, col*2].set_ylabel("Frequency", fontsize=8)
                axs[row, col*2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                axs[row, col*2].tick_params(axis="both", which="major", labelsize=7)
                
                # Calculate and display weight gradient distribution statistics
                weight_grad_mean = np.mean(weight_gradients)
                weight_grad_std = np.std(weight_gradients)
                axs[row, col*2].text(0.95, 0.95, 
                                    f"Mean: {weight_grad_mean:.4f}\nStd: {weight_grad_std:.4f}", 
                                    transform=axs[row, col*2].transAxes, 
                                    verticalalignment="top", 
                                    horizontalalignment="right",
                                    fontsize=7,
                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
                
                # Plot bias gradient distribution histogram
                axs[row, col*2+1].hist(bias_gradients, bins=50, edgecolor="black", alpha=0.7, color="lightgreen")
                axs[row, col*2+1].set_title(f"Layer {layer_index} - Bias Gradient Distribution", fontsize=10)
                axs[row, col*2+1].set_xlabel("Bias Gradient", fontsize=8)
                axs[row, col*2+1].set_ylabel("Frequency", fontsize=8)
                axs[row, col*2+1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                axs[row, col*2+1].tick_params(axis="both", which="major", labelsize=7)
                
                # Calculate and display bias gradient distribution statistics
                bias_grad_mean = np.mean(bias_gradients)
                bias_grad_std = np.std(bias_gradients)
                axs[row, col*2+1].text(0.95, 0.95, 
                                    f"Mean: {bias_grad_mean:.4f}\nStd: {bias_grad_std:.4f}", 
                                    transform=axs[row, col*2+1].transAxes, 
                                    verticalalignment="top", 
                                    horizontalalignment="right",
                                    fontsize=7,
                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        
        # Hide any unused subplots
        for row in range(num_rows):
            for col in range(4):
                if not axs[row, col].has_data():
                    axs[row, col].axis("off")
        
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def plot_loss_function(self):
        """
        Plotting the loss function for training and validation

        Parameters:
        None
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
    model.add_layer(4)
    model.add_layer(32, activation_function="elu", initialization_method="he_normal")
    model.add_layer(16, activation_function="tanh", initialization_method="xavier_normal")
    model.add_layer(2, activation_function="sigmoid", initialization_method="xavier_normal")

    # model.add_layer(2)
    # model.add_layer(2, activation_function='leaky_relu', activation_parameters={'alpha': 6666}, initialization_method='normal', mean=0, std=0.1) 
    # model.add_layer(1, activation_function='selu', activation_parameters={'alpha': 23123, 'lambda_': 123}, initialization_method='normal', mean=0, std=0.1)

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

    # x = np.array([[0,0], [0,1], [1,0], [1,1]])
    # y = np.array([[0], [0], [0], [1]])          # AND
    # y = np.array([[1], [1], [1], [0]])          # NAND
    # y = np.array([[0], [1], [1], [1]])          # OR
    # y = np.array([[1], [0], [0], [0]])          # NOR
    # y = np.array([[0], [1], [1], [0]])          # XOR
    # y = np.array([[1], [0], [0], [1]])          # XNOR

    x = np.array([[0,0,0,0],
                  [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0], 
                  [0,0,1,1], [0,1,0,1], [1,0,0,1], [0,1,1,0], [1,0,1,0], [1,1,0,0],
                  [0,1,1,1], [1,0,1,1], [1,1,0,1], [1,1,1,0], 
                  [1,1,1,1]])
    def xor(x1, x2, x3, x4):
        return (x1 ^ x2) ^ (x3 ^ x4)
    y = np.array([[xor(x1, x2, x3, x4), not xor(x1, x2, x3, x4)] for x1, x2, x3, x4 in x])

    # print("Input Data:\n", x)
    # print("True Labels:\n", y)
    # print("Predicted Labels:\n", model.predict(x))
    # print("Weights before training:\n", model.weights)
    # print("Biases before training:\n", model.biases)
    # print("Weights Gradient before training:\n", model.weights_gradient)
    # print("Biases Gradient before training:\n", model.biases_gradient)

    model.train(x, y, batch_size=1, learning_rate=0.1, epochs=10000, loss_function='MSE', verbose=1, error_threshold=0.0001, l1_lambda=0, l2_lambda=0)
    for i in range(len(y)):
        print(f"Predicted Labels for {x[i]}:", model.predict(x[i].reshape(1, -1)), "True Labels:", y[i])
    # model.plot_network_graph(visualize='weights')
    # model.plot_network_graph(visualize='gradients')
    # print("Weights after training:\n", model.weights)
    # print("Biases after training:\n", model.biases)
    # print("Weights Gradient after training:\n", model.weights_gradient)
    # print("Biases Gradient after training:\n", model.biases_gradient)
    # print(model.history)

    model.plot_network_graph(visualize="both")
    model.plot_weight_distribution([1, 2])
    model.plot_gradient_distribution([2, 3])

    # model.save("model.pkl")
    # loaded_model = FFNN.load("model.pkl")

if __name__ == "__main__":
    main()