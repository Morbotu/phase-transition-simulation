import cv2
import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import time

class MeltingFrontTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Melting Front Tracker")
        self.root.bind("<Configure>", self.on_resize)

        self.video_path = None
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.frame_interval = 60  # seconds
        self.current_image = None

        self.roi_mode = True
        self.roi = None
        self.lines_data = []
        self.scale = None
        self.first_measurement = None
        self.settings_file = None
        self.data_file = None
        self.window_width = self.root.winfo_width()
        self.window_height = self.root.winfo_height()
        self.orig_img_width = None
        self.orig_img_height = None
        self.frame_time = None
        self.scale_dist_px = tk.StringVar()
        self.scale_dist_px.set("Dist: - px")
        self.scale_text = tk.StringVar()
        self.scale_text.set("Scale: 1 mm/px")

        self.img_width = None
        self.img_height = None
        self.old_img_width = None
        self.old_img_height = None
        self.scale_lines = []
        self.scaleY = []

        self.image_on_canvas = None
        
        self.open_video()

        self.canvas = tk.Canvas(root)
        self.canvas.pack()
        
        self.load_frame()

        self.controls = tk.Frame(root)
        self.controls.pack()
        self.interface = []

        self.roi_description = tk.Label(self.controls, text="Press <Return> to accept ROI")
        self.roi_description.pack(side=tk.LEFT, padx=10)
        self.roi_square = None

        self.canvas.bind("<Button-1>", self.select_roi)
        self.canvas.bind("<B1-Motion>", self.select_roi)
        self.root.bind("<Return>", self.select_roi)
        self.root.bind("<Left>", self.prev_frame)
        self.root.bind("<Right>", self.next_frame)

    def on_resize(self, event):
        if self.window_width != self.root.winfo_width() or self.window_height != self.root.winfo_height():
            self.window_width = self.root.winfo_width()
            self.window_height = self.root.winfo_height()
            self.old_img_width = self.img_width
            self.old_img_height = self.img_height
            self.load_frame()

            if self.roi and self.roi_mode:
                self.roi = (
                    int(self.roi[0] * self.img_width / self.old_img_width),
                    int(self.roi[1] * self.img_height / self.old_img_height),
                    int(self.roi[2] * self.img_width / self.old_img_width),
                    int(self.roi[3] * self.img_height / self.old_img_height),
                )
                self.canvas.coords(self.roi_square, *self.roi)

    def select_roi(self, event):
        match int(event.type):
            case 2: # Enter
                if self.roi is None:
                    messagebox.showerror("Error", "No ROI selected.")
                    return
                self.close_roi()
                self.load_interface()
            case 4: # Mouse down
                self.roi = (event.x, event.y) * 2
            case 6: # Mouse drag
                self.roi = self.roi[:2] + (event.x, event.y)
        
        if self.roi_square is None:
            self.roi_square = self.canvas.create_rectangle(*self.roi, outline="red")
        else:
            self.canvas.coords(self.roi_square, *self.roi)

    def close_roi(self):
        self.roi_mode = False
        self.roi_description.pack_forget()

        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.root.unbind("<Return>")

        self.roi = (
            int(self.roi[0] * self.orig_img_width / self.img_width),
            int(self.roi[1] * self.orig_img_height / self.img_height),
            int(self.roi[2] * self.orig_img_width / self.img_width),
            int(self.roi[3] * self.orig_img_height / self.img_height),
        )
        self.canvas.delete(self.roi_square)

    def load_interface(self):
        self.set_scale_btn = tk.Button(self.controls, text="Set Scale", command=self.set_scale)
        self.set_scale_btn.pack(side=tk.LEFT)

        self.scale_label = tk.Label(self.controls, textvariable=self.scale_text)
        self.scale_label.pack(side=tk.LEFT, padx=10)

        self.prev_btn = tk.Button(self.controls, text="Previous", command=self.prev_frame)
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = tk.Button(self.controls, text="Next", command=self.next_frame)
        self.next_btn.pack(side=tk.LEFT)

        self.frame_time_label = tk.Label(self.controls, textvariable=self.frame_time)
        self.frame_time_label.pack(side=tk.LEFT, padx=10)

        self.total_time_label = tk.Label(self.controls, text=f"Max time: {self.total_frames / self.fps:.1f} s")
        self.total_time_label.pack(side=tk.LEFT, padx=10)

        self.interface = [
            self.set_scale_btn,
            self.scale_label,
            self.prev_btn,
            self.next_btn,
            self.frame_time_label,
            self.total_time_label
        ]

        self.load_frame()

    def close_set_scale(self):
        self.scale_description.pack_forget()
        self.scale_dist_label.pack_forget()
        for scale_line in self.scale_lines:
            self.canvas.delete(scale_line)
        self.scaleY = []
        self.scale_lines = []
        self.scale_dist_px.set("Dist: - px")

        self.canvas.unbind("<Button-1>")
        self.root.unbind("<Return>")

    def select_scale(self, event):
        match int(event.type):
            case 2: # Enter
                if len(self.scaleY) != 2:
                    messagebox.showerror("Error", "Please select two lines as reference.")
                    return
                real_dist = simpledialog.askfloat("Scale", "Enter real distance in mm between the two lines:")
                px_dist = abs(self.scaleY[1] - self.scaleY[0])
                self.scale = real_dist / px_dist * self.img_height / self.orig_img_height
                self.scale_text.set(f"Scale: {self.scale:.3f} mm/px")
                self.close_set_scale()
                self.load_interface()
            case 4: # Click
                if len(self.scaleY) >= 2:
                    for scale_line in self.scale_lines:
                        self.canvas.delete(scale_line)
                    self.scaleY = []
                    self.scale_lines = []
                    self.scale_dist_px.set("Dist: - px")
                    return
                self.scaleY.append(event.y)
                self.scale_lines.append(self.canvas.create_line(0, event.y, self.img_width, event.y, fill="red"))

                if len(self.scaleY) == 2:
                    self.scale_dist_px.set(f"Dist: {abs(self.scaleY[1] - self.scaleY[0])} px")

    def set_scale(self):
        for interfaceitem in self.interface:
            interfaceitem.pack_forget()

        self.scale_description = tk.Label(self.controls, text="Click to set scale reference. Press <Enter> to accept.")
        self.scale_description.pack(side=tk.LEFT, padx=10)

        self.scale_dist_label = tk.Label(self.controls, textvariable=self.scale_dist_px)
        self.scale_dist_label.pack(side=tk.LEFT, padx=10)

        self.canvas.bind("<Button-1>", self.select_scale)
        self.root.bind("<Return>", self.select_scale)

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not path:
            messagebox.showerror("Error", "Invalid path selected.")
            raise "Invalid path selected"
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_time = tk.StringVar()
        self.frame_time.set("Time: 0 s")

    def load_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to load frame.")
                raise "Failed to load frame"
            if not self.roi_mode:
                x1, y1, x2, y2 = self.roi
                frame = frame[y1:y2, x1:x2]
            self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image()
    
    def resize_image_to_fit_canvas(self, img, canvas_width, canvas_height):
        self.orig_img_width, self.orig_img_height = img.size
        img_ratio = self.orig_img_width / self.orig_img_height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            # Image is wider than canvas — fit to width
            self.img_width = canvas_width
            self.img_height = int(canvas_width / img_ratio)
        else:
            # Image is taller than canvas — fit to height
            self.img_height = canvas_height
            self.img_width = int(canvas_height * img_ratio)

        resized = img.resize((self.img_width, self.img_height), Image.LANCZOS)
        return resized

    def display_image(self):
        canvas_width = self.window_width
        canvas_height = self.window_height-30

        img = Image.fromarray(self.current_image.copy())
        img = self.resize_image_to_fit_canvas(img, canvas_width, canvas_height)

        img_tk = ImageTk.PhotoImage(img)
        self.canvas.config(width=img_tk.width(), height=img_tk.height())

        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=img_tk)
        
        self.canvas.image = img_tk

    def next_frame(self, event=None):
        step = round(self.fps * self.frame_interval)
        self.current_frame_idx = min(self.current_frame_idx + step, self.total_frames - 1)
        self.update_time()
        self.load_frame()

    def prev_frame(self, event=None):
        step = round(self.fps * self.frame_interval)
        self.current_frame_idx = max(self.current_frame_idx - step, 0)
        self.update_time()
        self.load_frame()

    def update_time(self):
        self.frame_time.set(f"Time: {self.current_frame_idx / self.fps:.1f} s")

if __name__ == "__main__":
    root = tk.Tk()
    app = MeltingFrontTracker(root)
    root.mainloop()