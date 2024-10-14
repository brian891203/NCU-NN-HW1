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

from modules.data_processor import DataProcessor
from modules.perceptron import Perceptron
from modules.plot_vis import *


class main_window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg='white')
        self.initUI()
        self.init_modules()

        self.canvas_widget_train: FigureCanvasTkAgg = None  # To hold training figure
        self.canvas_widget_test: FigureCanvasTkAgg = None   # To hold testing figure
        self.canvas_widget_loss: FigureCanvasTkAgg = None   # To hold loss curve figure
    
    def initUI(self):
        self.title('HW1 112526011')
        self.geometry('1200x680+10+0')
        self.minsize(1200, 680)  # 設置最小窗口大小
        self.maxsize(1200, 680)  # 設置最大窗口大小
        self.resizable(False, False)  # 禁止用戶調整窗口大小
        self.update_idletasks()  # 確保控件初始化完成

        self.Content()
        self.Bottom_Bar()

    def init_modules(self):
        self.model = Perceptron()
        self.data_processor = DataProcessor()
    
    def Content(self):
        # Content_mainframe contains other Content_subframes
        self.Content_mainframe = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_mainframe.pack(padx=0, pady=20)

        # sub1_component00 contains canvas1 for Training data
        self.sub1_component00 = tk.Frame(self.Content_mainframe, padx=10, pady=0, bg='white')
        self.sub1_component00.grid(row=0, column=0, padx=5, pady=0)
        self.canvas1 = tk.Canvas(self.sub1_component00, width=350, height=300, bg='white', highlightthickness=1, relief='solid')
        self.canvas1.grid(row=0, column=0)

        # sub1_component01 contains canvas2 for Testing data
        self.sub1_component01 = tk.Frame(self.Content_mainframe, padx=10, pady=0, bg='white')
        self.sub1_component01.grid(row=0, column=1, padx=5, pady=0)
        self.canvas2 = tk.Canvas(self.sub1_component01, width=350, height=300, bg='white', highlightthickness=1, relief='solid')
        self.canvas2.grid(row=0, column=0)

        # sub1_component02 contains canvas3 for Learning rate data
        self.sub1_component02 = tk.Frame(self.Content_mainframe, padx=10, pady=0, bg='white')
        self.sub1_component02.grid(row=0, column=2, padx=5, pady=0)
        self.canvas3 = tk.Canvas(self.sub1_component02, width=350, height=300, bg='white', highlightthickness=1, relief='solid')
        self.canvas3.grid(row=0, column=0)

    def Bottom_Bar(self):
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
        
    def open_file_event(self):
        # 開啟檔案並讀取資料
        file_path = filedialog.askopenfilename(initialdir="./", filetypes=[("Text Files", "*.txt *.TXT")])
        if file_path:
            data = self.data_processor.load_data(file_path)
            print(data)
            self.model.set_data(data)
            self.fig_train, self.fig_test, self.fig_loss = plt.figure(figsize=(3.5, 3)), plt.figure(figsize=(3.5, 3)), plt.figure(figsize=(3.5, 3))

        else:
            tk.messagebox.showerror("Error", "No file selected..")

    def train_mlp_event(self):
        """train the MLP"""
        if self.model.train_x is not None: 
            input_dim = self.model.train_x.shape[1]
            output_dim = self.model.train_y.shape[1]
            self.model.set_model(input_dim=input_dim, hidden_dim=16, output_dim=output_dim)
            learning_rate = float(self.learning_rate_entry.get())
            epochs = int(self.epoch_entry.get())
            self.model.train(learning_rate=learning_rate, epochs=epochs)

            # Plotting the figures
            if self.model.train_y.shape[1] == 2:
                self.fig_train, self.fig_test = plot_2D(self.fig_train, self.fig_test, self.model)
            else:
                self.fig_train, self.fig_test = plot_3D(self.fig_train, self.fig_test, self.model)
            self.fig_loss = plot_loss_curve(self.model, self.fig_loss)

            # Now, display figures on the canvas
            self.display_canvas(self.canvas1, self.fig_train, "canvas_widget_train")
            self.display_canvas(self.canvas2, self.fig_test, "canvas_widget_test")
            self.display_canvas(self.canvas3, self.fig_loss, "canvas_widget_loss")

    def display_canvas(self, canvas, fig, canvas_widget_name):
        """Display a figure on the given canvas."""
        # Clear previous canvas if exists
        if getattr(self, canvas_widget_name):
            getattr(self, canvas_widget_name).get_tk_widget().destroy()

        # Create a new FigureCanvasTkAgg and draw it
        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Store the new canvas widget
        setattr(self, canvas_widget_name, canvas_widget)


    
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