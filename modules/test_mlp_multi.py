import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_processor import DataProcessor
from matplotlib.colors import ListedColormap
from perceptron import Perceptron
from sklearn.decomposition import PCA


# 定義 2D 決策邊界繪圖函數
def plot_2D_decision_boundary(fig_train, fig_test, model):
    sns.set(style="whitegrid")  # 使用 seaborn 美化樣式

    fig_train.clf()
    fig_test.clf()
    
    ax_train = fig_train.add_subplot(111)
    ax_test = fig_test.add_subplot(111)

    # 訓練資料繪圖
    colors_train = np.argmax(model.train_y, axis=1)
    
    # 使用 seaborn 的配色方案繪製散點圖
    scatter_train = ax_train.scatter(model.train_x[:, 0], model.train_x[:, 1], 
                                     c=colors_train, cmap="flare",  # 使用 seaborn 的 coolwarm 顏色映射
                                     edgecolor='k', s=40, alpha=0.8, label='Train Data')
    
    # 繪製訓練集決策邊界
    x_min, x_max = model.train_x[:, 0].min() - 0.1, model.train_x[:, 0].max() + 0.1
    y_min, y_max = model.train_x[:, 1].min() - 0.1, model.train_x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = [model.forward(i) for i in x_in]
    y_pred = np.argmax(np.array(y_pred), axis=1)
    y_pred = y_pred.reshape(xx.shape)

    decision_surface_train = ax_train.contourf(xx, yy, y_pred, cmap="flare", alpha=0.5)

    num_classes = model.train_y.shape[1]  # 動態計算類別數
    class_labels_train = [f"Class {i}" for i in range(num_classes)]
    handles_train, labels_train = scatter_train.legend_elements()
    
    # 使用 seaborn 生成圖例
    ax_train.legend(handles_train, class_labels_train, title="Class Labels", frameon=True, fancybox=True, shadow=True)

    ax_train.set_title('Training Set Decision Boundary', fontsize=14)
    ax_train.set_xlabel('Feature 1')
    ax_train.set_ylabel('Feature 2')

    # 測試資料繪圖
    colors_test = np.argmax(model.test_y, axis=1)
    scatter_test = ax_test.scatter(model.test_x[:, 0], model.test_x[:, 1], 
                                   c=colors_test, cmap="flare", 
                                   edgecolor='k', s=40, alpha=0.8, label='Test Data')
    
    x_min, x_max = model.test_x[:, 0].min() - 0.1, model.test_x[:, 0].max() + 0.1
    y_min, y_max = model.test_x[:, 1].min() - 0.1, model.test_x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = [model.forward(i) for i in x_in]
    y_pred = np.argmax(np.array(y_pred), axis=1)
    y_pred = y_pred.reshape(xx.shape)

    decision_surface_test = ax_test.contourf(xx, yy, y_pred, cmap="flare", alpha=0.5)

    num_classes = model.test_y.shape[1]  # 動態計算類別數
    class_labels_test = [f"Class {i}" for i in range(num_classes)]
    handles_test, labels_test = scatter_test.legend_elements()
    
    ax_test.legend(handles_test, class_labels_test, title="Class Labels", frameon=True, fancybox=True, shadow=True)

    ax_test.set_title('Test Set Decision Boundary', fontsize=14)
    ax_test.set_xlabel('Feature 1')
    ax_test.set_ylabel('Feature 2')

    return fig_train, fig_test


# 定義 3D 決策邊界繪圖函數
def plot_3D_decision_boundary(fig_train, fig_test, model):
    print("plot_3D_decision_boundary......")
    
    # 清空之前的圖形
    fig_train.clf()
    fig_test.clf()

    # 創建 seaborn 的調色板來美化配色方案
    palette = sns.color_palette("crest", as_cmap=True)

    # 創建 3D 子圖
    ax_train = fig_train.add_subplot(111, projection='3d')
    ax_test = fig_test.add_subplot(111, projection='3d')
    
    # 訓練集資料處理
    x_train = model.train_x[:, :2]
    train_y = np.argmax(model.train_y, axis=1)
    
    # 繪製訓練集的資料點，使用 seaborn 調色板美化
    scatter_train = ax_train.scatter(x_train[:, 0], x_train[:, 1], train_y, 
                                     c=train_y, cmap=palette, 
                                     edgecolor='black', s=50, alpha=0.8)
    
    # 設置 3D 圖的視角
    ax_train.view_init(elev=30, azim=120)  # elev: 仰角, azim: 方位角
    
    # 計算訓練集決策邊界
    plot_surface_decision_boundary(ax_train, x_train, model)

    num_classes = model.train_y.shape[1]  # 動態計算類別數
    class_labels_train = [f"Class {i}" for i in range(num_classes)]
    handles_train, labels_train = scatter_train.legend_elements()
    ax_train.legend(handles_train, class_labels_train, title="Class Labels")

    # 測試集資料處理
    x_test = model.test_x[:, :2]
    test_y = np.argmax(model.test_y, axis=1)

    # 繪製測試集的資料點，使用 seaborn 調色板
    scatter_test = ax_test.scatter(x_test[:, 0], x_test[:, 1], test_y, 
                                   c=test_y, cmap=palette, 
                                   edgecolor='black', s=50, alpha=0.8)

    # 設置 3D 圖的視角
    ax_test.view_init(elev=30, azim=120)    

    # 計算測試集決策邊界
    plot_surface_decision_boundary(ax_test, x_test, model)
    num_classes = model.test_y.shape[1]  # 動態計算類別數
    class_labels_test = [f"Class {i}" for i in range(num_classes)]
    handles_test, labels_test = scatter_test.legend_elements()
    ax_test.legend(handles_test, class_labels_test, title="Class Labels")

    return fig_train, fig_test

# 繪製 3D 決策邊界的輔助函數
def plot_surface_decision_boundary(ax, data, model):
    palette = sns.color_palette("crest", as_cmap=True)

    # 計算 x, y 軸範圍
    x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    
    # 創建網格
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))  # 更高解析度
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # 使用模型對網格進行預測
    y_pred = [model.forward(point) for point in x_in]
    y_pred = np.argmax(np.array(y_pred), axis=1)
    y_pred = y_pred.reshape(xx.shape)

    # 繪製決策邊界，使用 seaborn 的 viridis 調色板
    ax.plot_surface(xx, yy, y_pred, cmap=palette, alpha=0.3)

# 主程式入口
if __name__ == '__main__':
    # 資料加載
    data_processor = DataProcessor()
    data = data_processor.load_data(r'C:\Users\User\Desktop\NN\HW\HW1\NCU-NN-HW1\data\extra\IRIS.TXT')
    print(data)

    # 使用原始多維資料進行感知機訓練（不降維）
    perceptron = Perceptron()
    perceptron.set_data(data)
    input_dim = perceptron.train_x.shape[1]
    output_dim = perceptron.train_y.shape[1]

    print("input_dim:", input_dim)
    print("output_dim:", output_dim)

    perceptron.set_model(input_dim=input_dim, hidden_dim=16, output_dim=output_dim)
    perceptron.train()
    
    # 預測訓練集和測試集的結果
    train_accuracy = perceptron.evaluate(perceptron.train_x, perceptron.train_y)
    test_accuracy = perceptron.evaluate(perceptron.test_x, perceptron.test_y)

    # 建立空白圖表以顯示結果
    fig_train, fig_test = plt.figure(), plt.figure()

    # 決定使用 2D 還是 3D 繪圖
    if perceptron.train_y.shape[1] == 2:
        fig_train, fig_test = plot_2D_decision_boundary(fig_train, fig_test, perceptron)
    else:
        fig_train, fig_test = plot_3D_decision_boundary(fig_train, fig_test, perceptron)

    plt.show()
