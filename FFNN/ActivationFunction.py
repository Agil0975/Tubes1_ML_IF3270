"""
This module contains the activation functions used in the FFNN model. 

List of activation functions
- Linear
- ReLU
- Sigmoid
- Hyperbolic Tangent (tanh)
- Softmax
"""

import numpy as np

class ActivationFunction:
    """
    Base class for activation functions.
    """

    def __linear(self, x: np.ndarray) -> np.ndarray:
        """
        Linear activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the linear activation function.
        """
        return x
    
    def __linear_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the linear activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the linear activation function.
        """
        return np.ones_like(x)
    
    def __relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the ReLU activation function.
        """
        return np.maximum(0, x)
    
    def __relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the ReLU activation function.
        """
        return np.where(x > 0, 1, 0)
    
    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    
    def __sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the sigmoid activation function.
        """
        sig = self.__sigmoid(x)
        return sig * (1 - sig)
    
    def __tanh(self, x: np.ndarray) -> np.ndarray:

        """
        Hyperbolic Tangent (tanh) activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the tanh activation function.
        """
        return np.tanh(x)
    
    def __tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the tanh activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the tanh activation function.
        """
        return 1 - np.tanh(x)**2
    
    def __softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the softmax activation function.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def __softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the softmax activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the softmax activation function.
        """
        s = self.__softmax(x)
        return s * (1 - s)
    
    def activation(self, x: np.ndarray, *, activation_function: str) -> np.ndarray:
        """
        Apply the specified activation function to the input.

        Parameters:
        x (np.ndarray): Input array.
        activation_function (str): Name of the activation function.

        Returns:
        np.ndarray: Output array after applying the activation function.
        """
        if activation_function == 'linear':
            return self.__linear(x)
        elif activation_function == 'relu':
            return self.__relu(x)
        elif activation_function == 'sigmoid':
            return self.__sigmoid(x)
        elif activation_function == 'tanh':
            return self.__tanh(x)
        elif activation_function == 'softmax':
            return self.__softmax(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_function}. Use 'linear', 'relu', 'sigmoid', 'tanh', or 'softmax'.")
        
    def activation_derivative(self, x: np.ndarray, *, activation_function: str) -> np.ndarray:
        """
        Apply the derivative of the specified activation function to the input.

        Parameters:
        x (np.ndarray): Input array.
        activation_function (str): Name of the activation function.

        Returns:
        np.ndarray: Output array after applying the derivative of the activation function.
        """
        if activation_function == 'linear':
            return self.__linear_derivative(x)
        elif activation_function == 'relu':
            return self.__relu_derivative(x)
        elif activation_function == 'sigmoid':
            return self.__sigmoid_derivative(x)
        elif activation_function == 'tanh':
            return self.__tanh_derivative(x)
        elif activation_function == 'softmax':
            return self.__softmax_derivative(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_function}. Use 'linear', 'relu', 'sigmoid', 'tanh', or 'softmax'.")
        

def main():
    # Example usage
    x = np.array([[1, 2], [-3, -4]])
    activation_function = ActivationFunction()
    
    print("Linear Activation:")
    print(activation_function.activation(x, activation_function='linear'))
    print(activation_function.activation_derivative(x, activation_function='linear'))
    
    print("\nReLU Activation:")
    print(activation_function.activation(x, activation_function='relu'))
    print(activation_function.activation_derivative(x, activation_function='relu'))
    
    print("\nSigmoid Activation:")
    print(activation_function.activation(x, activation_function='sigmoid'))
    print(activation_function.activation_derivative(x, activation_function='sigmoid'))
    
    print("\nTanh Activation:")
    print(activation_function.activation(x, activation_function='tanh'))
    print(activation_function.activation_derivative(x, activation_function='tanh'))
    
    print("\nSoftmax Activation:")
    print(activation_function.activation(x, activation_function='softmax'))
    print(activation_function.activation_derivative(x, activation_function='softmax'))

if __name__ == "__main__":
    main()