import tkinter as tk

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap


def plot_empty_on_canvas(canvas, fig, canvas_widget_name: str):
    """在 tkinter Canvas 上繪製空白圖表"""
    # 清空舊有圖形
    canvas.delete("all")

    # 繪製空白圖表
    ax = fig.gca()
    ax.set_title("No Data Available")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    # setattr(self, canvas_widget_name, canvas_widget)

# Define 2D decision boundary plotting function
def plot_2D(fig_train, fig_test, model):
    sns.set(style="whitegrid")

    fig_train.clf()
    fig_test.clf()
    
    ax_train = fig_train.add_subplot(111)
    ax_test = fig_test.add_subplot(111)

    # Plot training data
    colors_train = np.argmax(model.train_y, axis=1)
    
    # Use seaborn color palette to plot scatter plot
    scatter_train = ax_train.scatter(model.train_x[:, 0], model.train_x[:, 1], 
                                     c=colors_train, cmap="flare", 
                                     edgecolor='k', s=40, alpha=0.8, label='Train Data')
    
    # Plot decision boundary for training set
    x_min, x_max = model.train_x[:, 0].min() - 0.1, model.train_x[:, 0].max() + 0.1
    y_min, y_max = model.train_x[:, 1].min() - 0.1, model.train_x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = [model.forward(i) for i in x_in]
    y_pred = np.argmax(np.array(y_pred), axis=1)
    y_pred = y_pred.reshape(xx.shape)

    decision_surface_train = ax_train.contourf(xx, yy, y_pred, cmap="flare", alpha=0.5)

    num_classes = model.train_y.shape[1]  # Dynamically calculate number of classes
    class_labels_train = [f"Class {i}" for i in range(num_classes)]
    handles_train, labels_train = scatter_train.legend_elements()
    
    # Use seaborn to generate legend
    ax_train.legend(handles_train, class_labels_train, title="Class Labels", frameon=True, fancybox=True, shadow=True)

    ax_train.set_title('Training Set Decision Boundary', fontsize=14)
    ax_train.set_xlabel('Feature 1')
    ax_train.set_ylabel('Feature 2')

    # Plot test data
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

    num_classes = model.test_y.shape[1]  # Dynamically calculate number of classes
    class_labels_test = [f"Class {i}" for i in range(num_classes)]
    handles_test, labels_test = scatter_test.legend_elements()
    
    ax_test.legend(handles_test, class_labels_test, title="Class Labels", frameon=True, fancybox=True, shadow=True)

    ax_test.set_title('Test Set Decision Boundary', fontsize=14)
    ax_test.set_xlabel('Feature 1')
    ax_test.set_ylabel('Feature 2')

    return fig_train, fig_test

def plot_3D(fig_train, fig_test, model):
    print("plot_3D_decision_boundary......")
    
    # Clear previous figures
    fig_train.clf()
    fig_test.clf()

    # Create seaborn color palette for better aesthetics
    palette = sns.color_palette("crest", as_cmap=True)

    # Create 3D subplots
    ax_train = fig_train.add_subplot(111, projection='3d')
    ax_test = fig_test.add_subplot(111, projection='3d')
    
    # Process training data
    x_train = model.train_x[:, :2]
    train_y = np.argmax(model.train_y, axis=1)
    
    # Plot training data points using seaborn color palette
    scatter_train = ax_train.scatter(x_train[:, 0], x_train[:, 1], train_y, 
                                     c=train_y, cmap=palette, 
                                     edgecolor='black', s=50, alpha=0.8)
    
    # Set 3D plot view angle
    ax_train.view_init(elev=30, azim=120)  # elev: elevation angle, azim: azimuth angle
    
    # Compute decision boundary for training set
    plot_surface_decision_boundary(ax_train, x_train, model)

    num_classes = model.train_y.shape[1]  # Dynamically calculate number of classes
    class_labels_train = [f"Class {i}" for i in range(num_classes)]
    handles_train, labels_train = scatter_train.legend_elements()
    ax_train.legend(handles_train, class_labels_train, title="Class Labels")

    # Process test data
    x_test = model.test_x[:, :2]
    test_y = np.argmax(model.test_y, axis=1)

    # Plot test data points using seaborn color palette
    scatter_test = ax_test.scatter(x_test[:, 0], x_test[:, 1], test_y, 
                                   c=test_y, cmap=palette, 
                                   edgecolor='black', s=50, alpha=0.8)

    # Set 3D plot view angle
    ax_test.view_init(elev=30, azim=120)    

    # Compute decision boundary for test set
    plot_surface_decision_boundary(ax_test, x_test, model)
    num_classes = model.test_y.shape[1]  # Dynamically calculate number of classes
    class_labels_test = [f"Class {i}" for i in range(num_classes)]
    handles_test, labels_test = scatter_test.legend_elements()
    ax_test.legend(handles_test, class_labels_test, title="Class Labels")

    return fig_train, fig_test

# Helper function to plot 3D decision boundary
def plot_surface_decision_boundary(ax, data, model):
    palette = sns.color_palette("crest", as_cmap=True)

    # Compute x, y axis range
    x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    
    # Create mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))  # Higher resolution
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # Use model to predict on the grid
    y_pred = [model.forward(point) for point in x_in]
    y_pred = np.argmax(np.array(y_pred), axis=1)
    y_pred = y_pred.reshape(xx.shape)

    # Plot decision boundary using seaborn's viridis color palette
    ax.plot_surface(xx, yy, y_pred, cmap=palette, alpha=0.3)

def plot_loss_curve(model, fig_loss):
    ax_loss = fig_loss.add_subplot(111)
    ax_loss.clear()
    ax_loss.plot(range(1,len(model.loss_record)+1),model.loss_record, label='Training Loss')
    
    ax_loss.set_xlabel('epochs')
    ax_loss.set_ylabel('loss')
    ax_loss.grid(True)
    ax_loss.legend()

    return fig_loss