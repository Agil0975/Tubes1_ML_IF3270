"""
This file contains the loss functions used in the FFNN model.

List of loss functions
- Mean Squared Error
- Binary Cross Entropy
- Categorical Cross Entropy
"""

import numpy as np
from typing import Tuple, Union

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
        return np.mean(np.square(y_true - y_pred))
    
    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    
    def categorical_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    
    