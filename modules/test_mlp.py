import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 讀取資料集
data = np.loadtxt(r'C:\Users\User\Desktop\NN\HW\HW1\NCU-NN-HW1\data\basic\2CloseS.txt')
X = data[:, :2]  # 前兩列為特徵值（x 和 y 坐標）
y = data[:, 2]   # 最後一列為標籤（1 或 2）

# 將標籤轉換為感知機可處理的格式（例如：1 和 -1）
y = np.where(y == 1, 1, -1)

# 隨機將資料分為 2/3 訓練資料和 1/3 測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 感知機類別
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        num_samples, num_features = X.shape
        # 初始化權重和偏置
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 訓練感知機
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # 預測輸出
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_function(linear_output)

                # 更新規則
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)

# 建立感知機並訓練
perceptron = Perceptron(learning_rate=0.1, epochs=100)
perceptron.train(X_train, y_train)

# 預測訓練集和測試集的結果
y_train_pred = perceptron.predict(X_train)
y_test_pred = perceptron.predict(X_test)

# 計算準確率
train_accuracy = np.mean(y_train_pred == y_train) * 100
test_accuracy = np.mean(y_test_pred == y_test) * 100

print(f"訓練集準確率: {train_accuracy:.2f}%")
print(f"測試集準確率: {test_accuracy:.2f}%")

# 視覺化訓練結果與測試結果
def plot_decision_boundary(X, y, model, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap=plt.cm.Paired, edgecolor='k')
    ax.legend(*scatter.legend_elements(), title="Class")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Perceptron Model - Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

# 畫訓練集資料點與分類結果
ax[0].set_title("Training Set")
plot_decision_boundary(X_train, y_train, perceptron, ax[0])

# 畫測試集資料點與分類結果
ax[1].set_title("Test Set")
plot_decision_boundary(X_test, y_test, perceptron, ax[1])

plt.show()
