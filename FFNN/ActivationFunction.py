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

    def linear(self, x: np.ndarray) -> np.ndarray:
        """
        Linear activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the linear activation function.
        """
        return x
    
    def linear_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the linear activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the linear activation function.
        """
        return np.ones_like(x)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the ReLU activation function.
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the ReLU activation function.
        """
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the sigmoid activation function.
        """
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:

        """
        Hyperbolic Tangent (tanh) activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the tanh activation function.
        """
        return np.tanh(x)
    
    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the tanh activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the tanh activation function.
        """
        return 1 - np.tanh(x)**2
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the softmax activation function.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the softmax activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the softmax activation function.
        """
        s = self.softmax(x)
        return s * (1 - s)