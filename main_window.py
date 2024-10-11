import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from sklearn.decomposition import PCA

from modules.mlp import *


class main_window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg='white')
        self.initUI()
        self.init_mlp()
        self.Top_Bar()
        self.Content()

        self.img = None
        self.canvas_widget_train: FigureCanvasTkAgg = None  # FigureCanvasTkAgg object for containing the fig_train
        self.canvas_widget_test: FigureCanvasTkAgg = None
        
    def initUI(self):
        self.title('HW1 112526011')
        self.geometry('1200x680+10+0')
    
    def init_mlp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.perceptron = None
    
    def Top_Bar(self):
        self.Top_Bar_mainframe = tk.Frame(self, padx=10, pady=20, bg='white')
        self.Top_Bar_mainframe.pack(padx=0, pady=0)

        self.open_file_button = tk.Button(self.Top_Bar_mainframe, text='File', command=self.open_file_event, width=15, height=2, bg='white')
        self.open_file_button.grid(row=0, column=0, padx=10, pady=0)

        self.flip_img_button = tk.Button(self.Top_Bar_mainframe, text='Training', command=self.train_mlp_event, width=15, height=2, bg='white')
        self.flip_img_button.grid(row=0, column=1, padx=10, pady=0)

        label_learning_rate = tk.Label(self.Top_Bar_mainframe, text="Learning rate :", bg='white', anchor=tk.W)
        label_learning_rate.grid(row=1, column=1, padx=10, pady=3, sticky=tk.W)
        self.learning_rate_entry = tk.Entry(self.Top_Bar_mainframe, width=3, highlightthickness=1)
        self.learning_rate_entry.grid(row=1, column=1, padx=10, pady=3, sticky=tk.E)

        label_epoch = tk.Label(self.Top_Bar_mainframe, text="Epoch :", bg='white', anchor=tk.W)
        label_epoch.grid(row=2, column=1, padx=10, pady=3, sticky=tk.W)
        self.epoch_entry = tk.Entry(self.Top_Bar_mainframe, width=3, highlightthickness=1)
        self.epoch_entry.grid(row=2, column=1, padx=10, pady=3, sticky=tk.E)

        self.show_histogram_button = tk.Button(self.Top_Bar_mainframe, text='Histogram', command=self.show_histogram_event, width=15, height=2, bg='white')
        self.show_histogram_button.grid(row=0, column=2, padx=10, pady=0)

        self.save_img_button = tk.Button(self.Top_Bar_mainframe, text='Save', command=self.save_img_event, width=15, height=2, bg='white')
        self.save_img_button.grid(row=0, column=3, padx=10, pady=0)

    def Content(self):
        # Content_mainframe contains other Content_subframes
        self.Content_mainframe = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_mainframe.pack(padx=0, pady=20)

        # Content_subframe1 contains Content_frame_00 and Content_frame_01
        self.Content_subframe1 = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_subframe1.pack(padx=0, pady=0)

        # sub1_component00 contains canvas1 for Training data
        self.sub1_component00 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.sub1_component00.grid(row=0, column=0, padx=20, pady=0)

        self.canvas1 = tk.Canvas(self.sub1_component00, width=500, height=400, bg='white', highlightthickness=1, relief='solid')
        self.canvas1.grid(row=0, column=0)

        # sub1_component01 contains canvas2 for Testing data
        self.sub1_component01 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.sub1_component01.grid(row=0, column=1, padx=20, pady=0)

        self.canvas2 = tk.Canvas(self.sub1_component01, width=500, height=400, bg='white', highlightthickness=1, relief='solid')
        self.canvas2.grid(row=0, column=0)

        #Content_subframe1 contains Content_frame_00 and Content_frame_01
        self.Content_subframe2 = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_subframe2.pack(padx=0, pady=0)

        # sub2_component00 contains canva_histogram
        self.sub2_component00 = tk.Frame(self.Content_subframe2, padx=10, pady=20, bg='white')
        self.sub2_component00.grid(row=0, column=0, padx=10, pady=0)

        self.canvas_histogram = tk.Canvas(self.sub2_component00, width=500, height=200, bg='white', highlightthickness=0)
        self.canvas_histogram.grid(row=0, column=0)

    def open_file_event(self):
        # 開啟檔案並讀取資料
        file_path = filedialog.askopenfilename(initialdir="./", filetypes=[("Text Files", "*.txt *.TXT")])
        if file_path:
            # 使用 data_loader 函數載入資料
            self.X_train, self.X_test, self.y_train, self.y_test = data_loader(file_path)

            # 使用 PCA 將資料降到 2 維（僅為了顯示用）
            self.pca = PCA(n_components=2)
            self.X_train_pca = self.pca.fit_transform(self.X_train)
            self.X_test_pca = self.pca.transform(self.X_test)

            # 將訓練集的降維資料繪製到 canvas1
            self.fig_train, self.ax_train = plt.subplots(1, 1, figsize=(5, 4))
            self.plot_data_on_canvas(
                self.canvas1, 
                self.fig_train, 
                self.ax_train,
                "canvas_widget_train",
                self.X_train_pca, 
                self.y_train, 
                "Training Data")

            # 將測試集的降維資料繪製到 canvas2
            self.fig_test, self.ax_test = plt.subplots(1, 1, figsize=(5, 4))
            self.plot_data_on_canvas(
                self.canvas2, 
                self.fig_test, 
                self.ax_test, 
                "canvas_widget_test",
                self.X_test_pca, 
                self.y_test,
                "Testing Data")

        else:
            tk.messagebox.showerror("Error", "No file selected..")

    def plot_data_on_canvas(self, canvas, fig, ax, canvas_widget_name:str, X, y, title):
        """將資料繪製到 tkinter Canvas 中"""
        # 清空舊有圖形
        canvas.delete("all")
        if getattr(self, canvas_widget_name) is not None:
            print("check")
            getattr(self, canvas_widget_name).get_tk_widget().destroy()

        # 訓練集資料顯示
        scatter_train = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['lightblue', 'orange']), edgecolor='k')
        ax.legend(*scatter_train.legend_elements(), title="Class")
        # ax.set_xlabel('Principal Component 1')
        # ax.set_ylabel('Principal Component 2')
        ax.set_title(f"{title}")

        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        setattr(self, canvas_widget_name, canvas_widget)

    def train_mlp_event(self):
        """train the MLP"""
        # try:
        if self.X_train is None or self.y_train is None:
            tk.messagebox.showerror("Error", "No training data found.")
            return

        # 訓練感知機
        self.perceptron = Perceptron(
            learning_rate=float(self.learning_rate_entry.get()),
            epochs=int(self.epoch_entry.get()))
        self.perceptron.train(self.X_train, self.y_train)

        y_train_pred = self.perceptron.predict(self.X_train)
        y_test_pred = self.perceptron.predict(self.X_test)

        # plot the results of the training
        self.plot_decision_boundary(self.ax_train, self.perceptron.weights, "canvas_widget_train")
        self.plot_decision_boundary(self.ax_test, self.perceptron.weights, "canvas_widget_test")

        # 計算訓練集與測試集的準確率
        train_accuracy = np.mean(y_train_pred == self.y_train) * 100
        test_accuracy = np.mean(y_test_pred == self.y_test) * 100

        print(f"Train accuracy : {train_accuracy:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")

        # except Exception as e:
        #     tk.messagebox.showerror("Error", "Error occurred while plotting decision boundary.")
        #     print("Error occurred while plotting decision boundary:", e)

    def plot_decision_boundary(self, ax, weights, canvas_widget_name:str):
        """繪製決策邊界"""
        # 畫出決策邊界（使用降維後的二維資料來計算）
        x_min, x_max = self.X_train_pca[:, 0].min(), self.X_train_pca[:, 0].max()
        y_min, y_max = self.X_train_pca[:, 1].min(), self.X_train_pca[:, 1].max()
        # y_min -= abs((y_max - y_min)/2)
        # y_max += abs((y_max - y_min)/2)
        print("x_min: ", x_min)
        print("x_max: ", x_max)
        print("y_min: ", y_min)
        print("y_max: ", y_max)

        x_values = np.linspace(x_min, x_max, 100)

        # 獲取訓練好的權重
        weights = weights
        if len(weights) == 3:
            w0, w1, w2 = weights
            y_values = -(w0 + w1 * x_values) / w2
        else:
            pca_weights = self.pca.components_
            w0 = weights[0]
            w1, w2 = pca_weights @ weights[1:]
            y_values = -(w0 + w1 * x_values) / w2

        valid_indices = (y_values >= y_min) & (y_values <= y_max)
        x_values_filtered = x_values[valid_indices]
        y_values_filtered = y_values[valid_indices]
        
        x_max, x_min = x_values_filtered.max(), x_values_filtered.min()
        y_max, y_min = y_values_filtered.max(), y_values_filtered.min()
        print("x_min_filtered: ", x_min)
        print("x_max_filtered: ", x_max)
        print("y_min_filtered: ", y_min)
        print("y_max_filtered: ", y_max)

        # 在原有的圖上疊加決策邊界
        ax.plot(x_values_filtered, y_values_filtered, 'k--')  # 黑色虛線表示決策邊界
        canvas_widget = getattr(self, canvas_widget_name)
        canvas_widget.draw()

        # tk.messagebox.showinfo("Training Complete", "Perceptron training completed!")
    
    def save_img_event(self):
        if self.img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")])
            if file_path:
                save_image = self.img
                save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2RGB)
                save_image = cv2.resize(save_image, (self.original_width, self.original_height))
                cv2.imwrite(file_path, save_image)
                tk.messagebox.showinfo("Success", f"Image saved successfully at {file_path}")
            else:
                tk.messagebox.showerror("Error", "No file save path selected.")
        else:
            tk.messagebox.showerror("Error", "Please open an image first.")

    def show_histogram_event(self):
        pass

if __name__ == '__main__':
    app = main_window()
    app.mainloop()