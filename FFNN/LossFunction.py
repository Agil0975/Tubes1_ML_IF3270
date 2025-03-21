"""
This module contains the loss functions used in the FFNN model.

List of loss functions
- Mean Squared Error
- Binary Cross Entropy
- Categorical Cross Entropy
"""

import numpy as np

class LossFunction:
    """
    Base class for loss functions.
    """

    def MSE(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error loss function.

        Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        float: Mean Squared Error.
        """
        return np.sum(np.square(y_true - y_pred), axis=1).mean() # Average over the batch size
    
    def MSE_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of Mean Squared Error loss function.

        Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        np.ndarray: Derivative of Mean Squared Error.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]  # Average over the batch size

    def BCE(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross Entropy loss function.

        Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        float: Binary Cross Entropy.
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def BCE_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of Binary Cross Entropy loss function.

        Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        np.ndarray: Derivative of Binary Cross Entropy.
        """
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - ((y_pred - y_true) / (y_pred * (1 - y_pred))) / y_true.size
    
    def CCE(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Categorical Cross Entropy loss function.

        Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        float: Categorical Cross Entropy.
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def CCE_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of Categorical Cross Entropy loss function.

        Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        np.ndarray: Derivative of Categorical Cross Entropy.
        """
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) / y_true.shape[0]

def main():
    # Example usage
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.8, 0.2], [0.1, 0.9]])
    
    loss_fn = LossFunction()
    
    print("Mean Squared Error:", loss_fn.MSE(y_true, y_pred))
    print("Binary Cross Entropy:", loss_fn.BCE(y_true, y_pred))
    print("Categorical Cross Entropy:", loss_fn.CCE(y_true, y_pred))
    print("MSE Derivative:", loss_fn.MSE_derivative(y_true, y_pred))
    print("BCE Derivative:", loss_fn.BCE_derivative(y_true, y_pred))
    print("CCE Derivative:", loss_fn.CCE_derivative(y_true, y_pred))

if __name__ == "__main__":
    main()