import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


class Gaussian_noise_window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg='white')
        self.initUI()
        self.Top_Bar()
        self.Content()

        self.img = None
        self.sigma = None
        
    def initUI(self):
        self.title('AIP 61275008H')
        self.geometry('880x680+100+0')
    
    def Top_Bar(self):
        self.Top_Bar_mainframe = tk.Frame(self, padx=10, pady=20, bg='white')
        self.Top_Bar_mainframe.pack(padx=0, pady=0)

        self.open_file_button = tk.Button(self.Top_Bar_mainframe, text='File', command=self.open_file_event, width=15, height=2, bg='white')
        self.open_file_button.grid(row=0, column=0, padx=10, pady=0)

        self.show_noise_button = tk.Button(self.Top_Bar_mainframe, text='Gaussian noise', command=self.generate_gaussian_noise, width=15, height=2, bg='white')
        self.show_noise_button.grid(row=0, column=1, padx=10, pady=0)

        label = tk.Label(self.Top_Bar_mainframe, text="Input sigma:", bg='white', anchor=tk.W)
        label.grid(row=1, column=1, padx=10, pady=3, sticky=tk.W)

        self.show_noise_entry = tk.Entry(self.Top_Bar_mainframe, width=5, highlightthickness=1)
        self.show_noise_entry.grid(row=1, column=1, padx=10, pady=3, sticky=tk.E)

        self.show_histogram_button = tk.Button(self.Top_Bar_mainframe, text='Histogram', command=self.show_histogram_event, width=15, height=2, bg='white')
        self.show_histogram_button.grid(row=0, column=2, padx=10, pady=0)

        self.save_img_button = tk.Button(self.Top_Bar_mainframe, text='Save', command=self.save_img_event, width=15, height=2, bg='white')
        self.save_img_button.grid(row=0, column=3, padx=10, pady=0)


    def open_file_event(self):
        file_path = filedialog.askopenfilename(initialdir="./", filetypes=[("Image Files", "*.jpg *.bmp *.ppm")])
        if file_path:
            self.img = cv2.imread(file_path)
            self.original_height, self.original_width, self.channels = self.img.shape
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            #show the image on canva
            self.img = cv2.resize(self.img, (210, 210))
            self.Tkimg = Image.fromarray(self.img)
            self.Tkimg = ImageTk.PhotoImage(self.Tkimg)
            self.canvas1.create_image(105, 105, anchor=tk.CENTER, image=self.Tkimg)
    
        else:
            tk.messagebox.showerror("Error", "No image selected.")
    
    def save_img_event(self):
        if hasattr(self, 'noisy_image'):
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")])
            if file_path:
                save_image = cv2.resize(self.noisy_image, (self.original_height, self.original_width))
                cv2.imwrite(file_path, save_image)
                tk.messagebox.showinfo("Success", f"Image saved successfully at {file_path}")
            else:
                tk.messagebox.showerror("Error", "No file save path selected.")
        else:
            tk.messagebox.showerror("Error", "Please generate noise first.")

    def show_histogram_event(self):
        if hasattr(self, 'histogram_fig') and self.histogram_fig is not None:
            self.histogram_fig.get_tk_widget().destroy()
        if hasattr(self, 'histogram_fig1') and self.histogram_fig1 is not None:
            self.histogram_fig1.get_tk_widget().destroy()
        if hasattr(self, 'histogram_fig2') and self.histogram_fig2 is not None:
            self.histogram_fig2.get_tk_widget().destroy()

        if self.img is not None and hasattr(self, 'noisy_image') and hasattr(self, 'noise'):
            #self.img =================================================================
            pixels = np.array(self.img.flatten())
            sns.histplot(pixels, bins=256, kde=False, color='skyblue', alpha=0.7, edgecolor='black')
            
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)

            plt.title('Images Histogram',fontsize=7)
            plt.gcf().set_size_inches(2.75, 2.75)

            self.histogram_fig = FigureCanvasTkAgg(plt.gcf(), master=self.canvas_histogram)
            self.histogram_fig.draw()
            self.histogram_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            plt.close()

            #self.noise ================================================================
            pixels = np.array(self.noise.flatten())
            values, counts = np.unique(pixels, return_counts=True)
            values = values[np.argmax(counts)]  
            pixels = pixels[pixels!=values]          
            sns.histplot(pixels, bins=255, kde=False, color='skyblue', alpha=1, edgecolor='black')
            
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)

            plt.title('Images Histogram',fontsize=7)
            plt.ylabel('')
            plt.gcf().set_size_inches(2.75, 2.75)

            self.histogram_fig1 = FigureCanvasTkAgg(plt.gcf(), master=self.canva_noise_histogram)
            self.histogram_fig1.draw()
            self.histogram_fig1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            plt.close()

            #self.noisy_image ================================================================
            pixels = np.array(self.noisy_image.flatten())
            sns.histplot(pixels, bins=256, kde=False, color='skyblue', alpha=0.7, edgecolor='black')
            
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            plt.title('Images Histogram',fontsize=7)

            plt.gcf().set_size_inches(2.75, 2.75)

            self.histogram_fig2 = FigureCanvasTkAgg(plt.gcf(), master=self.canva_noise_image_histogram)
            self.histogram_fig2.draw()
            self.histogram_fig2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            plt.close()
        
        elif self.img is None:
            tk.messagebox.showerror("Error", "Please open an image first.")
        else:
            tk.messagebox.showerror("Error", "Please input sigma first.")
    
    def generate_gaussian_noise(self):
        user_input_sigma = self.show_noise_entry.get()
        if user_input_sigma:
            try:
                if user_input_sigma.isdigit():
                    self.sigma = int(user_input_sigma)
                elif isinstance(float(user_input_sigma), float):
                    self.sigma = float(user_input_sigma)
                if self.sigma <= 0:
                    self.sigma = None
                    tk.messagebox.showerror("Error", "Please input sigma over 0.")
            except:
                self.sigma = None
                tk.messagebox.showerror("Error", "Please input sigma properly.")
        else:
            tk.messagebox.showerror("Error", "Please input sigma first.")

        if self.img is not None and (isinstance(self.sigma, int) or isinstance(self.sigma, float)):
            G = 256
            self.noisy_image = np.copy(self.img)  #self.noisy_image --> 798 * 800
            self.noise = np.zeros((self.noisy_image.shape[0], self.noisy_image.shape[1], 3), dtype=np.int16)
            self.noise_true_value = np.zeros((self.noisy_image.shape[0], self.noisy_image.shape[1]), dtype=np.int16)
            for x in range(self.noisy_image.shape[0]):      #798 rows
                for y in range(0, self.noisy_image.shape[1], 2):  #800 cols

                    r1, r2 = np.random.rand(2)
                    z1 = self.sigma * np.cos(2 * np.pi * r2) * np.sqrt(-2 * np.log(r1))
                    z2 = self.sigma * np.sin(2 * np.pi * r2) * np.sqrt(-2 * np.log(r1))
                    f1 = self.img[x, y][0] + z1
                    f2 = self.img[x, y+1][0] + z2
                
                    if f1 < 0:
                        self.noisy_image[x, y][:] = 0
                    elif f1 > G - 1:
                        self.noisy_image[x, y][:] = G - 1
                    else:
                        self.noisy_image[x, y][:] = f2

                    if f2 < 0:
                        self.noisy_image[x, y+1][:] = 0
                    elif f2 > G - 1:
                        self.noisy_image[x, y+1][:] = G - 1
                    else:
                        self.noisy_image[x, y+1][:] = f2

                    self.noise_true_value[x, y] = f1 - self.img[x, y][0]
                    self.noise_true_value[x, y+1] = f2 - self.img[x, y+1][0]

            min_value = np.min(self.noise_true_value)
            max_value = np.max(self.noise_true_value)
            self.noise = (self.noise_true_value - min_value) / (max_value - min_value) * 255
            self.noise = self.noise.astype(np.uint8)

            self.noise_show = cv2.resize(self.noise, (210, 210))
            self.noise_show = Image.fromarray(self.noise_show)
            self.noise_show = ImageTk.PhotoImage(self.noise_show)

            self.noisy_image_show = cv2.resize(self.noisy_image, (210, 210))
            self.noisy_image_show = Image.fromarray(self.noisy_image_show)
            self.noisy_image_show = ImageTk.PhotoImage(self.noisy_image_show)

            self.canvas_noise.create_image(105, 105, anchor=tk.CENTER, image=self.noise_show)
            self.canvas_noise_image.create_image(105, 105, anchor=tk.CENTER, image=self.noisy_image_show)

        elif self.img is None:
            tk.messagebox.showerror("Error", "Please open an image first.")
        

    def Content(self):
        # Content_mainframe contains other Content_subframes
        self.Content_mainframe = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_mainframe.pack(padx=0, pady=0)

        #Content_subframe1 contains Content_frame_00 and Content_frame_01
        self.Content_subframe1 = tk.Frame(self, padx=12, pady=0, bg='white')
        self.Content_subframe1.pack(padx=0, pady=0)

        # Content_frame00 contains canvas1
        self.Content_frame_00 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.Content_frame_00.grid(row=0, column=0, padx=30, pady=0)

        self.canvas1 = tk.Canvas(self.Content_frame_00, width=210, height=210, bg='white', highlightthickness=0)
        self.canvas1.grid(row=0, column=0)

        # Content_frame01 contains canvas_noise
        self.Content_frame_01 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.Content_frame_01.grid(row=0, column=1, padx=30, pady=0)

        self.canvas_noise = tk.Canvas(self.Content_frame_01, width=210, height=210, bg='white', highlightthickness=0)
        self.canvas_noise.grid(row=0, column=0)

        # Content_frame02 contains canvas_noise_image
        self.Content_frame_02 = tk.Frame(self.Content_subframe1, padx=10, pady=0, bg='white')
        self.Content_frame_02.grid(row=0, column=2, padx=30, pady=0)

        self.canvas_noise_image = tk.Canvas(self.Content_frame_02, width=210, height=210, bg='white', highlightthickness=0)
        self.canvas_noise_image.grid(row=0, column=0)

        #Content_subframe2 contains Content_frame10 and Content_frame10 and Content_frame11 and Content_frame12
        self.Content_subframe2 = tk.Frame(self, padx=10, pady=0, bg='white')
        self.Content_subframe2.pack(padx=0, pady=0)

        # Content_frame10 contains canva_histogram
        self.Content_frame_10 = tk.Frame(self.Content_subframe2, padx=10, pady=20, bg='white')
        self.Content_frame_10.grid(row=1, column=0, padx=0, pady=0)

        self.canvas_histogram = tk.Canvas(self.Content_frame_10, width=275, height=275, bg='white', highlightthickness=0)
        self.canvas_histogram.grid(row=0, column=0)

        # Content_frame11 contains canva_noise_histogram
        self.Content_frame_11 = tk.Frame(self.Content_subframe2, padx=10, pady=20, bg='white')
        self.Content_frame_11.grid(row=1, column=1, padx=0, pady=0)

        self.canva_noise_histogram = tk.Canvas(self.Content_frame_11, width=275, height=275, bg='white', highlightthickness=0)
        self.canva_noise_histogram.grid(row=0, column=0)

        # Content_frame12 contains canva_noise_image_histogram
        self.Content_frame_12 = tk.Frame(self.Content_subframe2, padx=10, pady=20, bg='white')
        self.Content_frame_12.grid(row=1, column=2, padx=0, pady=0)

        self.canva_noise_image_histogram = tk.Canvas(self.Content_frame_12, width=275, height=275, bg='white', highlightthickness=0)
        self.canva_noise_image_histogram.grid(row=0, column=0)

if __name__ == '__main__':
    app = Gaussian_noise_window()
    app.mainloop()