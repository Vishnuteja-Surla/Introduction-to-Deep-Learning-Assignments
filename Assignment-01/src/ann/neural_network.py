import numpy as np
from activations import ReLU, Sigmoid, Tanh, Softmax
from neural_layer import NeuralLayer
from objective_functions import MSE, CrossEntropy

ACTIVATIONS = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax
}

LOSS_FUNCTIONS = {
    'mean_squared_error': MSE,
    'cross_entropy': CrossEntropy
}

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """

        # Reading the CLI arguments
        self.loss_fn = LOSS_FUNCTIONS[cli_args.loss]()
        self.optimizer = cli_args.optimizer
        self.lr = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        self.num_layers = cli_args.num_layers
        self.hidden_size = cli_args.hidden_size
        self.activation = ACTIVATIONS[cli_args.activation]()
        self.weight_init = cli_args.weight_init
        # self.optim     (Yet to be implemented)

        # Creating the Neural Network
        self.layers = []
        input_size = 784
        num_classes = 10
        for i in range(0, self.num_layers):
            layer = NeuralLayer(input_size, self.hidden_size[i], self.activation, self.weight_init)
            input_size = self.hidden_size[i]
            self.layers.append(layer)
        layer = NeuralLayer(input_size, num_classes, ACTIVATIONS['softmax'](), self.weight_init)
        self.layers.append(layer)

    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        A = X
        for i in range(0, len(self.layers)):
            A = self.layers[i].forward(A)
        return A
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        dA = self.loss_fn.backward(y_true, y_pred)
        for i in range(len(self.layers)-1, -1, -1):
            dA = self.layers[i].backward(dA)
        return dA
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optim.update(self.layers)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            
            shuffled_indices = np.random.permutation(num_samples)
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]
            epoch_loss = 0

            for i in range(0, num_samples, batch_size):
                X_batch = X_train_shuffled[i : i+batch_size]
                y_batch = y_train_shuffled[i : i+batch_size]

                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update_weights()

                epoch_loss += self.loss_fn.forward(y_batch, y_pred)

            if epoch%10 == 0:
                print(f"Epoch-{epoch+1}: Average Loss = {epoch_loss / (num_samples/batch_size)}")
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred = self.forward(X)
        loss_value = self.loss_fn.forward(y, y_pred)
        predicted_classes = np.argmax(y_pred, axis=1)
        actual_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predicted_classes == actual_classes)

        return loss_value, accuracy