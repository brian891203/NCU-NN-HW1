import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


class Gray_scale_histogram_window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg='white')
        self.initUI()
        self.Top_Bar()
        self.Content()

        self.img = None
        
    def initUI(self):
        self.title('AIP 61275008H')
        self.geometry('880x680+100+0')
    
    def Top_Bar(self):
        self.Top_Bar_mainframe = tk.Frame(self, padx=10, pady=20, bg='white')
        self.Top_Bar_mainframe.pack(padx=0, pady=0)

        self.open_file_button = tk.Button(self.Top_Bar_mainframe, text='File', command=self.open_file_event, width=15, height=2, bg='white')
        self.open_file_button.grid(row=0, column=0, padx=10, pady=0)

        self.flip_img_button = tk.Button(self.Top_Bar_mainframe, text='Flip', command=self.flip_img_event, width=15, height=2, bg='white')
        self.flip_img_button.grid(row=0, column=1, padx=10, pady=0)

        self.show_histogram_button = tk.Button(self.Top_Bar_mainframe, text='Histogram', command=self.show_histogram_event, width=15, height=2, bg='white')
        self.show_histogram_button.grid(row=0, column=2, padx=10, pady=0)

        self.save_img_button = tk.Button(self.Top_Bar_mainframe, text='Save', command=self.save_img_event, width=15, height=2, bg='white')
        self.save_img_button.grid(row=0, column=3, padx=10, pady=0)


    def open_file_event(self):
        file_path = filedialog.askopenfilename(initialdir="./", filetypes=[("Image Files", "*.jpg *.bmp *.ppm")])
        if file_path:
            self.img = cv2.imread(file_path)
            self.original_height, self.original_width, self.channels = self.img.shape
            self.img = cv2.resize(self.img, (200, 200))

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.Tkimg = Image.fromarray(self.img)
            self.Tkimg = ImageTk.PhotoImage(self.Tkimg)
            self.canvas1.create_image(100, 100, anchor=tk.CENTER, image=self.Tkimg)
    
        else:
            tk.messagebox.showerror("Error", "No image selected..")
    
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

    def Content(self):
        # Content_mainframe contains other Content_subframes
        self.Content_mainframe = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_mainframe.pack(padx=0, pady=20)

        #Content_subframe1 contains Content_frame_00 and Content_frame_01
        self.Content_subframe1 = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_subframe1.pack(padx=0, pady=0)

        # Content_frame00 contains canvas1
        self.Content_frame_00 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.Content_frame_00.grid(row=0, column=0, padx=20, pady=0)

        self.canvas1 = tk.Canvas(self.Content_frame_00, width=200, height=200, bg='white', highlightthickness=0)
        self.canvas1.grid(row=0, column=0)

        # Content_frame01 contains canvas2
        self.Content_frame_01 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.Content_frame_01.grid(row=0, column=1, padx=20, pady=0)

        self.canvas2 = tk.Canvas(self.Content_frame_01, width=200, height=200, bg='white', highlightthickness=0)
        self.canvas2.grid(row=0, column=0)

        #Content_subframe1 contains Content_frame_00 and Content_frame_01
        self.Content_subframe2 = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_subframe2.pack(padx=0, pady=0)

        # Content_frame10 contains canva_histogram
        self.Content_frame_10 = tk.Frame(self.Content_subframe2, padx=10, pady=20, bg='white')
        self.Content_frame_10.grid(row=0, column=1, padx=10, pady=0)

        self.canvas_histogram = tk.Canvas(self.Content_frame_10, width=500, height=200, bg='white', highlightthickness=0)
        self.canvas_histogram.grid(row=0, column=0)

if __name__ == '__main__':
    app = Gray_scale_histogram_window()
    app.mainloop()