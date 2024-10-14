import numpy as np


class Perceptron:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.weights = None
        self.biases = None
        self.loss_record = []

    def set_data(self, data):
        self.train_x, self.train_y, self.test_x, self.test_y = data

    def set_model(self, input_dim, output_dim, hidden_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights = {
            'W1': np.random.randn(self.hidden_dim, self.input_dim),
            'W2': np.random.randn(self.output_dim, self.hidden_dim)
        }
        self.biases = {
            'b1': np.zeros(self.hidden_dim),
            'b2': np.zeros(self.output_dim)
        }

    def forward(self, x):
        z1 = np.dot(self.weights['W1'], x) + self.biases['b1']
        self.a1 = self.sigmoid(z1)
        z2 = np.dot(self.weights['W2'], self.a1) + self.biases['b2']
        a2 = self.softmax(z2)
        return a2

    def backward(self, x, y_true, y_pred):
        loss = self.categorical_cross_entropy(y_true, y_pred)
        dz2 = y_pred - y_true
        dW2 = np.outer(dz2, self.a1)
        db2 = dz2
        dz1 = np.dot(self.weights['W2'].T, dz2) * self.a1 * (1 - self.a1)
        dW1 = np.outer(dz1, x)
        db1 = dz1
        return loss, {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def train(self, epochs=100, learning_rate=0.01):
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for i in range(self.train_x.shape[0]):
                y_pred = self.forward(self.train_x[i])
                loss, gradients = self.backward(self.train_x[i], self.train_y[i], y_pred)
                total_loss += loss
                self._update_parameters(gradients, learning_rate)
            total_loss /= self.train_x.shape[0]
            self.loss_record.append(total_loss)
            print(f"Epoch {epoch}, Loss: {total_loss}")

    def evaluate(self, test_x, test_y):
        y_pred = [np.argmax(self.forward(x)) for x in test_x]
        test_y_ed = [np.argmax(y) for y in test_y]
        accuracy = np.mean(np.array(test_y_ed) == np.array(y_pred))
        print("Accuracy：", accuracy)
        return accuracy

    def _update_parameters(self, gradients, learning_rate):
        for param in self.weights.keys():
            self.weights[param] -= learning_rate * gradients[param]
        for param in self.biases.keys():
            self.biases[param] -= learning_rate * gradients[param]

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return Perceptron.sigmoid(x) * (1 - Perceptron.sigmoid(x))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))