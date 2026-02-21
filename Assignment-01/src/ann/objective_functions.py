import numpy as np

class Loss:
    """Base class for all loss functions."""
    
    def forward(self, y_true, y_pred):
        """Calculates the scalar loss value."""
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        """Calculate the gradient of loss w.r.t the predictions (dA)."""
        raise NotImplementedError
    
class MSE(Loss):
    """Class for Mean-Squared Error implementation."""
    
    def forward(self, y_true, y_pred):
        b = y_true.shape[0]    # Batch Size
        return 1/b * (np.sum((y_true - y_pred)**2))

    def backward(self, y_true, y_pred):
        b = y_true.shape[0]    # Batch Size
        return 2/b * (y_pred - y_true)


class CrossEntropy(Loss):
    """Class for Cross Entropy Loss implementation."""
    
    def forward(self, y_true, y_pred):
       b = y_true.shape[0]  # Batch Size
       return -1/b * np.sum(y_true * np.log(np.maximum(y_pred, 1e-15)))   # Lower Bounding probability to avoid log(0) 

    def backward(self, y_true, y_pred):
        b = y_true.shape[0] # Batch Size
        return 1/b * (y_pred - y_true)  # Assuming we are going to use Softmax Activation function