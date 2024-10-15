import numpy as np


class Perceptron:
    def __init__(self):
        self.train_inputs = None
        self.train_labels = None
        self.test_inputs = None
        self.test_labels = None
        self.weights = None
        self.biases = None
        self.loss_record = []

    def set_data(self, data):
        self.train_inputs, self.train_labels, self.test_inputs, self.test_labels = data

    def set_model(self, input_dim, output_dim, hidden_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights = {
            'weights_hidden': np.random.randn(self.hidden_dim, self.input_dim),
            'weights_output': np.random.randn(self.output_dim, self.hidden_dim)
        }
        self.biases = {
            'bias_hidden': np.zeros(self.hidden_dim),
            'bias_output': np.zeros(self.output_dim)
        }

    def forward(self, input_data):
        hidden_layer_input = np.dot(self.weights['weights_hidden'], input_data) + self.biases['bias_hidden']
        self.hidden_layer_activation = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(self.weights['weights_output'], self.hidden_layer_activation) + self.biases['bias_output']
        output_layer_activation = self.softmax(output_layer_input)
        return output_layer_activation

    def backward(self, input_data, true_labels, predicted_labels):
        loss = self.categorical_cross_entropy(true_labels, predicted_labels)
        output_layer_error = predicted_labels - true_labels
        gradients_weights_output = np.outer(output_layer_error, self.hidden_layer_activation)
        gradients_bias_output = output_layer_error
        hidden_layer_error = np.dot(self.weights['weights_output'].T, output_layer_error) * self.hidden_layer_activation * (1 - self.hidden_layer_activation)
        gradients_weights_hidden = np.outer(hidden_layer_error, input_data)
        gradients_bias_hidden = hidden_layer_error
        return loss, {'weights_hidden': gradients_weights_hidden, 'bias_hidden': gradients_bias_hidden, 'weights_output': gradients_weights_output, 'bias_output': gradients_bias_output}

    def train(self, epochs=100, learning_rate=0.01):
        self.loss_record = []
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for i in range(self.train_inputs.shape[0]):
                predicted_labels = self.forward(self.train_inputs[i])
                loss, gradients = self.backward(self.train_inputs[i], self.train_labels[i], predicted_labels)
                total_loss += loss
                self._update_parameters(gradients, learning_rate)
            total_loss /= self.train_inputs.shape[0]
            self.loss_record.append(total_loss)
            print(f"Epoch {epoch}, Loss: {total_loss}")

    def evaluate(self, test_inputs, test_labels):
        predicted_labels = [np.argmax(self.forward(input_data)) for input_data in test_inputs]
        true_labels = [np.argmax(label) for label in test_labels]
        accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
        print("Accuracy：", accuracy)
        return accuracy

    def _update_parameters(self, gradients, learning_rate):
        for param in self.weights.keys():
            self.weights[param] -= learning_rate * gradients[param]
        for param in self.biases.keys():
            self.biases[param] -= learning_rate * gradients[param]

    @staticmethod
    def relu(input_data):
        return np.maximum(0, input_data)

    @staticmethod
    def sigmoid(input_data):
        return 1 / (1 + np.exp(-input_data))

    @staticmethod
    def sigmoid_derivative(input_data):
        return Perceptron.sigmoid(input_data) * (1 - Perceptron.sigmoid(input_data))

    @staticmethod
    def softmax(input_data):
        e_x = np.exp(input_data - np.max(input_data))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def categorical_cross_entropy(true_labels, predicted_labels):
        epsilon = 1e-15
        predicted_labels = np.clip(predicted_labels, epsilon, 1 - epsilon)
        return -np.sum(true_labels * np.log(predicted_labels))