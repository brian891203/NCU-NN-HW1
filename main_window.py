import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from modules.mlp import *


class main_window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg='white')
        self.initUI()
        self.Top_Bar()
        self.Content()

        self.img = None
        
    def initUI(self):
        self.title('HW1 112526011')
        self.geometry('1200x680+10+0')
    
    def Top_Bar(self):
        self.Top_Bar_mainframe = tk.Frame(self, padx=10, pady=20, bg='white')
        self.Top_Bar_mainframe.pack(padx=0, pady=0)

        self.open_file_button = tk.Button(self.Top_Bar_mainframe, text='File', command=self.open_file_event, width=15, height=2, bg='white')
        self.open_file_button.grid(row=0, column=0, padx=10, pady=0)

        self.flip_img_button = tk.Button(self.Top_Bar_mainframe, text='Training', command=self.flip_img_event, width=15, height=2, bg='white')
        self.flip_img_button.grid(row=0, column=1, padx=10, pady=0)

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

        self.canvas1 = tk.Canvas(self.sub1_component00, width=400, height=400, bg='white', highlightthickness=1, relief='solid')
        self.canvas1.grid(row=0, column=0)

        # sub1_component01 contains canvas2 for Testing data
        self.sub1_component01 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.sub1_component01.grid(row=0, column=1, padx=20, pady=0)

        self.canvas2 = tk.Canvas(self.sub1_component01, width=400, height=400, bg='white', highlightthickness=1, relief='solid')
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
            pca = PCA(n_components=2)
            self.X_train_pca = pca.fit_transform(self.X_train)
            self.X_test_pca = pca.transform(self.X_test)

            # 將訓練集的降維資料繪製到 canvas1
            self.fig_train, self.ax_train = plt.subplots(1, 1, figsize=(5, 4))
            self.plot_data_on_canvas(
                self.canvas1, 
                self.fig_train, 
                self.ax_train, 
                self.X_train_pca, 
                self.y_train, 
                "Training Data (PCA)")

            # 將測試集的降維資料繪製到 canvas2
            self.fig_test, self.ax_test = plt.subplots(1, 1, figsize=(5, 4))
            self.plot_data_on_canvas(
                self.canvas2, 
                self.fig_test, 
                self.ax_test, 
                self.X_test_pca, 
                self.y_test,
                "Testing Data (PCA)")

        else:
            tk.messagebox.showerror("Error", "No file selected..")

    def plot_data_on_canvas(self, canvas, fig, ax, X, y, title):
        """將資料繪製到 tkinter Canvas 中"""
        # 清空舊有圖形
        canvas.delete("all")

        ax.clear()

        # 訓練集資料顯示
        scatter_train = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['lightblue', 'orange']), edgecolor='k')
        ax.legend(*scatter_train.legend_elements(), title="Class")
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(f"{title}")

        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack()
    
    def flip_img_event(self):
        if self.img is not None:
            self.img = cv2.flip(self.img, 0)  # 1 for horizontal flip
            self.Tkflipped_image = Image.fromarray(self.img)
            self.Tkflipped_image = ImageTk.PhotoImage(self.Tkflipped_image)        
            self.canvas2.create_image(100, 100, anchor=tk.CENTER, image=self.Tkflipped_image)
        else:
            tk.messagebox.showerror("Error", "Please open an image first.")

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
        if hasattr(self, 'histogram_fig') and self.histogram_fig is not None:
            self.histogram_fig.get_tk_widget().destroy()

        if self.img is not None:
            pixels = np.array(self.img.flatten())
            sns.histplot(pixels, bins=256, kde=False, color='skyblue', alpha=0.7, edgecolor='black')

            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)

            plt.title('Images Histogram',fontsize=10)
            plt.xlabel('Intensity',fontsize=10)
            plt.ylabel('Frequency',fontsize=10)

            plt.gcf().set_size_inches(5, 2)

            self.histogram_fig = FigureCanvasTkAgg(plt.gcf(), master=self.canvas_histogram)
            self.histogram_fig.draw()
            self.histogram_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            plt.close()
        
        else:
            tk.messagebox.showerror("Error", "Please open an image first.")

if __name__ == '__main__':
    app = main_window()
    app.mainloop()