import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split


def data_loader():
    # 讀取資料集
    data = np.loadtxt(r'C:\Users\User\Desktop\NN\HW\HW1\NCU-NN-HW1\data\basic\2Ccircle1.txt')
    print(data)

    X = data[:, :2]  # 前兩列為特徵值（x 和 y 坐標）
    y = data[:, -1]   # 最後一列為標籤（1 或 2）

    # 將標籤轉換為感知機可處理的格式（例如：1 和 -1）
    y = np.where(y == 1, 1, -1) # do this when using the single perceptron --> 單層只能分兩類

    # 隨機將資料分為 2/3 訓練資料和 1/3 測試資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

# 感知機類別
class Perceptron:
    def __init__(self, learning_rate:float=0.01, epochs:int=1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y, weights=None, bias=-1):
        num_samples, num_features = X.shape
        print(X.shape)

        # 初始化權重
        self.bias = bias
        if weights is None:
            self.weights = np.zeros(num_features)
        else: 
            self.weights = weights
        self.weights = np.insert(self.weights, 0, self.bias, axis=None)
        self.weights = self.weights.astype(np.float64)

        # 訓練感知機
        for epoch in range(self.epochs):
            print(f"======= Epoch {epoch} ========")
            for idx, x_i in enumerate(X):
                print(f"n = {idx} ===")
                x_i = np.insert(x_i, 0, -1, axis=None)
                x_i = x_i.astype(np.float64)

                print("weights : ", self.weights)
                print(f"x_{idx} : ", x_i)

                # 預測輸出
                linear_output = np.dot(x_i, self.weights)
                y_pred = self.activation_function(linear_output)

                print("y_pred : ", y_pred)
                print("y_truth : ", y[idx])

                # 更新規則
                print("result of prediction ===")
                if y_pred == y[idx]:
                    print("prediction is correct")
                    print("weights", self.weights)
                    print("bias", self.bias)
                    continue
                else:
                    print("prediction is not correct")
                    update = self.learning_rate * y[idx]  # == learning_rate * yi(Ground truth of y)
                    self.weights += update * x_i
                    # self.bias -= update  # 老師的 slide 沒有 update bias

                    print("update", update)
                    print("weights", self.weights)
                    print("bias", self.bias)

    def activation_function(self, v):
        return np.where(v >= 0, 1, -1)

    def predict(self, X):
        print("===== Starting prediction... =====")
        print("input_X : \n", X)

        X = np.insert(X, 0, -1, axis=1)
        linear_output = np.dot(X, self.weights)

        return self.activation_function(linear_output)
    
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_loader()

    # 建立感知機並訓練
    perceptron = Perceptron(learning_rate=0.8, epochs=100)
    perceptron.train(X_train, y_train, weights=np.array([0, 1]))

    # 預測訓練集和測試集的結果
    print("===== train_accuracy_result =====")
    print("X_train : \n", X_train)
    print("y_train : \n", y_train)
    y_train_pred = perceptron.predict(X_train)
    print("y_train_pred : \n", y_train_pred)

    print("===== test_accuracy_result =====")
    print("X_test : \n", X_test)
    print("y_test : \n", y_test)
    y_test_pred = perceptron.predict(X_test)
    print("y_test_pred : \n", y_test_pred)

    # 計算準確率
    train_accuracy = np.mean(y_train_pred == y_train) * 100
    test_accuracy = np.mean(y_test_pred == y_test) * 100

    print(f"訓練集準確率: {train_accuracy:.2f}%")
    print(f"測試集準確率: {test_accuracy:.2f}%")

    # 使用訓練好的權重來繪製決策邊界
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Perceptron Model - Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # 畫訓練集資料點
    scatter = ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(['lightblue', 'orange']), edgecolor='k')
    ax[0].legend(*scatter.legend_elements(), title="Class")
    ax[0].set_xlabel('Feature 1')
    ax[0].set_ylabel('Feature 2')
    ax[0].set_title("Training Set")

    # 畫決策邊界
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 100)
    # 透過決策邊界公式計算 x_2 的值
    w0, w1, w2 = perceptron.weights  # 假設 weights = [bias, w1, w2]
    y_values = (w0 - w1 * x_values) / w2

    ax[0].plot(x_values, y_values, 'k--')  # 黑色虛線表示決策邊界

    # 畫測試集資料點
    scatter = ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(['lightblue', 'orange']), edgecolor='k')
    ax[1].legend(*scatter.legend_elements(), title="Class")
    ax[1].set_xlabel('Feature 1')
    ax[1].set_ylabel('Feature 2')
    ax[1].set_title("Test Set")

    # 在測試集圖中畫出相同的決策邊界
    ax[1].plot(x_values, y_values, 'k--')  # 黑色虛線表示決策邊界

    plt.show()  