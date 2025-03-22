"""
This module contains the activation functions used in the FFNN model. 

List of activation functions
- Linear
- ReLU
- Sigmoid
- Hyperbolic Tangent (tanh)
- Softmax
- Leaky ReLU
- ELU (Exponential Linear Unit)
- SELU (Scaled Exponential Linear Unit)
- Swish
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
    
    def __leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.
        alpha (float): Slope of the function for negative values.

        Returns:
        np.ndarray: Output array after applying the leaky ReLU activation function.
        """
        # print('alpha', alpha)
        return np.where(x > 0, x, alpha * x)
    
    def __leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Derivative of the leaky ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.
        alpha (float): Slope of the function for negative values.

        Returns:
        np.ndarray: Derivative of the leaky ReLU activation function.
        """
        # print('alpha', alpha)
        return np.where(x > 0, 1, alpha)
    
    def __elu(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Exponential Linear Unit (ELU) activation function.

        Parameters:
        x (np.ndarray): Input array.
        alpha (float): Scaling factor for negative values.

        Returns:
        np.ndarray: Output array after applying the ELU activation function.
        """
        # print('alpha', alpha)
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def __elu_derivative(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Derivative of the ELU activation function.

        Parameters:
        x (np.ndarray): Input array.
        alpha (float): Scaling factor for negative values.

        Returns:
        np.ndarray: Derivative of the ELU activation function.
        """
        # print('alpha', alpha)
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    def __selu(self, x: np.ndarray, alpha: float = 1.6733, lambda_: float = 1.0507) -> np.ndarray:
        """
        Scaled Exponential Linear Unit (SELU) activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the SELU activation function.
        """
        # print('alpha', alpha, 'lambda_', lambda_)
        return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def __selu_derivative(self, x: np.ndarray, alpha: float = 1.6733, lambda_: float = 1.0507) -> np.ndarray:
        """
        Derivative of the SELU activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the SELU activation function.
        """
        # print('alpha', alpha, 'lambda_', lambda_)
        return lambda_ * np.where(x > 0, 1, alpha * np.exp(x))
    
    def __swish(self, x: np.ndarray) -> np.ndarray:
        """
        Swish activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the Swish activation function.
        """
        return x / (1 + np.exp(-x))
    
    def __swish_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the Swish activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Derivative of the Swish activation function.
        """
        sig = self.__sigmoid(x)
        return sig + x * sig * (1 - sig)
    
    def activation(self, x: np.ndarray, *, activation_type: str, **kwargs: dict) -> np.ndarray:
        """
        Apply the specified activation function to the input.

        Parameters:
        x (np.ndarray): Input array.
        activation_type (str): Name of the activation function.
        kwargs (dict): Additional parameters for the activation function.

        Returns:
        np.ndarray: Output array after applying the activation function.
        """
        if activation_type == 'linear':
            return self.__linear(x)
        elif activation_type == 'relu':
            return self.__relu(x)
        elif activation_type == 'sigmoid':
            return self.__sigmoid(x)
        elif activation_type == 'tanh':
            return self.__tanh(x)
        elif activation_type == 'softmax':
            return self.__softmax(x)
        elif activation_type == 'leaky_relu':
            return self.__leaky_relu(x, alpha=kwargs.get('alpha', 0.01))
        elif activation_type == 'elu':
            return self.__elu(x, alpha=kwargs.get('alpha', 1.0))
        elif activation_type == 'selu':
            return self.__selu(x, alpha=kwargs.get('alpha', 1.6733), lambda_=kwargs.get('lambda_', 1.0507))
        elif activation_type == 'swish':
            return self.__swish(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}. Use 'linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu', 'selu', or 'swish'.")

    def activation_derivative(self, x: np.ndarray, *, activation_type: str, **kwargs: dict) -> np.ndarray:
        """
        Apply the derivative of the specified activation function to the input.

        Parameters:
        x (np.ndarray): Input array.
        activation_type (str): Name of the activation function.
        kwargs (dict): Additional parameters for the activation function.

        Returns:
        np.ndarray: Output array after applying the derivative of the activation function.
        """
        if activation_type == 'linear':
            return self.__linear_derivative(x)
        elif activation_type == 'relu':
            return self.__relu_derivative(x)
        elif activation_type == 'sigmoid':
            return self.__sigmoid_derivative(x)
        elif activation_type == 'tanh':
            return self.__tanh_derivative(x)
        elif activation_type == 'softmax':
            return self.__softmax_derivative(x)
        elif activation_type == 'leaky_relu':
            return self.__leaky_relu_derivative(x, alpha=kwargs.get('alpha', 0.01))
        elif activation_type == 'elu':
            return self.__elu_derivative(x, alpha=kwargs.get('alpha', 1.0))
        elif activation_type == 'selu':
            return self.__selu_derivative(x, alpha=kwargs.get('alpha', 1.6733), lambda_=kwargs.get('lambda_', 1.0507))
        elif activation_type == 'swish':
            return self.__swish_derivative(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}. Use 'linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu', 'selu', or 'swish'.")

def main():
    # Example usage
    x = np.array([[1, 2], [-3, -4]])
    activation_function = ActivationFunction()
    
    print("Linear Activation:")
    print(activation_function.activation(x, activation_type='linear'))
    print(activation_function.activation_derivative(x, activation_type='linear'))
    
    print("\nReLU Activation:")
    print(activation_function.activation(x, activation_type='relu'))
    print(activation_function.activation_derivative(x, activation_type='relu'))
    
    print("\nSigmoid Activation:")
    print(activation_function.activation(x, activation_type='sigmoid'))
    print(activation_function.activation_derivative(x, activation_type='sigmoid'))
    
    print("\nTanh Activation:")
    print(activation_function.activation(x, activation_type='tanh'))
    print(activation_function.activation_derivative(x, activation_type='tanh'))
    
    print("\nSoftmax Activation:")
    print(activation_function.activation(x, activation_type='softmax'))
    print(activation_function.activation_derivative(x, activation_type='softmax'))

    print("\nLeaky ReLU Activation:")
    print(activation_function.activation(x, activation_type='leaky_relu', alpha=0.01))
    print(activation_function.activation_derivative(x, activation_type='leaky_relu', alpha=0.01))

    print("\nELU Activation:")
    print(activation_function.activation(x, activation_type='elu', alpha=1.0))
    print(activation_function.activation_derivative(x, activation_type='elu', alpha=1.0))

    print("\nSELU Activation:")
    print(activation_function.activation(x, activation_type='selu', alpha=1.6733, lambda_=1.0507))
    print(activation_function.activation_derivative(x, activation_type='selu', alpha=1.6733, lambda_=1.0507))

    print("\nSwish Activation:")
    print(activation_function.activation(x, activation_type='swish'))
    print(activation_function.activation_derivative(x, activation_type='swish'))

    print("\nFalse Activation:")
    try:
        print(activation_function.activation(x, activation_type='false'))
    except ValueError as e:
        print(e)

    print("\nFalse Activation Derivative:")
    try:
        print(activation_function.activation_derivative(x, activation_type='false'))
    except ValueError as e:
        print(e)

    dictio = {'type': 'selu', 'param': {'alpha': 1.6733, 'lambda_': 1.0507}}
    a = (dictio['type'], dictio['param'])
    print("\nSELU Activation with Dictionary:")
    print(activation_function.activation(x, activation_type=a[0], **a[1]))

if __name__ == "__main__":
    main()