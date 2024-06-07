# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 11:17:09 2024

@author: eduardo
"""
import tkinter as tk
from tkinter import ttk
class Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600")
        self.title('AstroIMP')
        self.resizable(0, 0)
        # configure the grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.create_widgets()
    def create_widgets(self):
        # input
        username_label = ttk.Label(self, text="Image directory to analyze:")
        username_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        username_entry = ttk.Entry(self)
        username_entry.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)
        # output
        password_label = ttk.Label(self, text="Image directory to save results:")
        password_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        password_entry = ttk.Entry(self)
        password_entry.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)
        # analyze
        login_button = ttk.Button(self, text="Analyze")
        login_button.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)
if __name__ == "__main__":
    app = Gui()
    app.mainloop()