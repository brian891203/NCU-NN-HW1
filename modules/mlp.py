import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# 資料加載函數
def data_loader(filepath):
    # 讀取資料集
    data = np.loadtxt(filepath)
    print("Loaded data:\n", data)

    X = data[:, :-1]  # 最後一列為標籤，前面所有列為特徵值
    y = data[:, -1]   # 最後一列為標籤（1 或 2）

    # 將標籤轉換為感知機可處理的格式（例如：1 和 -1）
    y = np.where(y == 1, 1, -1)  # 單層感知機只能分兩類時，將標籤轉換為 1 和 -1

    # 隨機將資料分為 2/3 訓練資料和 1/3 測試資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

# 感知機類別
class Perceptron:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y, weights=None, bias=-1):
        num_samples, num_features = X.shape

        # 初始化權重與偏置項
        self.bias = bias
        if weights is None:
            self.weights = np.zeros(num_features)
        else:
            self.weights = weights

        # 插入偏置項到權重中
        self.weights = np.insert(self.weights, 0, self.bias, axis=None)
        self.weights = self.weights.astype(np.float64)

        # 訓練感知機
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # 插入偏置項到輸入向量
                x_i = np.insert(x_i, 0, -1, axis=None)
                x_i = x_i.astype(np.float64)

                # 預測輸出
                linear_output = np.dot(x_i, self.weights)
                y_pred = self.activation_function(linear_output)

                # 若預測錯誤則更新權重
                if y_pred != y[idx]:
                    update = self.learning_rate * y[idx]
                    self.weights += update * x_i  # 更新權重

    def activation_function(self, v):
        return np.where(v >= 0, 1, -1)

    def predict(self, X):
        # 在特徵前面插入偏置項 -1
        X = np.insert(X, 0, -1, axis=1)
        linear_output = np.dot(X, self.weights)
        return self.activation_function(linear_output)
    

# 畫圖與結果顯示函數
def plot_decision_boundary(perceptron, X_train, y_train, X_test, y_test, canvas, ax):
    # 使用 PCA 將訓練與測試資料降至 2 維（僅為了可視化顯示）
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 計算決策邊界範圍並建立網格
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # 預測網格點的分類結果
    grid_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])  # 還原為原始維度資料
    Z = perceptron.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 畫決策邊界與資料點
    ax.clear()
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['lightblue', 'orange']))
    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=ListedColormap(['lightblue', 'orange']), edgecolor='k')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title("Perceptron Decision Boundary")

    # 畫決策邊界
    x_values = np.linspace(x_min, x_max, 100)
    
    if len(perceptron.weights) == 3:
        # 二維資料情況下，直接使用 w0, w1, w2
        w0, w1, w2 = perceptron.weights
        y_values = -(w0 + w1 * x_values) / w2
    else:
        # 多維資料情況，選擇 PCA 中的兩個主成分方向的權重作為 w1, w2
        pca_weights = pca.components_
        w0 = perceptron.weights[0]
        w1, w2 = pca_weights @ perceptron.weights[1:]
        y_values = -(w0 + w1 * x_values) / w2

    ax.plot(x_values, y_values, 'k--')  # 黑色虛線表示決策邊界
    canvas.draw()
