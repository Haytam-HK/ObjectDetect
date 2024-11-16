import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
import numpy as np
import tensorflow as tf
import cv2
import threading
import json
import os


class AdvancedObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Slinding window")
        self.master.geometry("1200x800")
        self.master.configure(bg="#f0f0f0")

        self.model = self.load_model()
        self.create_widgets()
        self.create_menu()

        self.current_image = None
        self.processed_image = None
        self.detections = []

        self.load_config()

    def load_model(self):
        return tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

    def create_widgets(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.main_frame = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.main_frame, padding="10")
        self.right_frame = ttk.Frame(self.main_frame, padding="10")

        self.main_frame.add(self.left_frame, weight=3)
        self.main_frame.add(self.right_frame, weight=1)

        # Left frame - Image display
        self.canvas = tk.Canvas(self.left_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right frame - Controls and results
        self.create_control_panel()

    def create_control_panel(self):
        # Buttons
        self.btn_load = ttk.Button(self.right_frame, text="SÉLECTIONNER", command=self.load_image, width=20)
        self.btn_load.pack(pady=5)

        self.btn_detect = ttk.Button(self.right_frame, text="Détecter", command=self.detect_objects, width=20)
        self.btn_detect.pack(pady=5)

        # Confidence threshold slider
        self.confidence_var = tk.DoubleVar(value=0.5)
        ttk.Label(self.right_frame, text="Confidence Threshold:").pack(pady=(10, 0))
        self.confidence_slider = ttk.Scale(self.right_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                                           variable=self.confidence_var, command=self.update_confidence)
        self.confidence_slider.pack(fill=tk.X, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.right_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress.pack(pady=10)

        # Results text area
        self.results_text = tk.Text(self.right_frame, height=20, width=30, font=('Arial', 10), wrap=tk.WORD)
        self.results_text.pack(pady=10, fill=tk.BOTH, expand=True)

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences", command=self.show_preferences)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Dark Mode", command=self.toggle_dark_mode)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            self.current_image = Image.open(file_path).convert("RGB")
            self.display_image(self.current_image)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Image chargée avec succès")

    def display_image(self, image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.photo)

    def detect_objects(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Veuillez d'abord charger une image")
            return

        self.progress.start()
        self.btn_detect.config(state=tk.DISABLED)

        # Run detection in a separate thread
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        image_array = np.array(self.current_image)
        self.detections = self.sliding_window_detection(image_array)
        self.master.after(0, self.update_results)

    def sliding_window_detection(self, image):
        detections = []
        (h, w) = image.shape[:2]

        for scale in [1.0, 0.75, 0.5]:
            resized = cv2.resize(image, (int(w * scale), int(h * scale)))

            for (x, y, window) in self.sliding_window(resized, step_size=64, window_size=(128, 128)):
                if window.shape[0] != 128 or window.shape[1] != 128:
                    continue

                window = cv2.resize(window, (224, 224))
                window = tf.keras.applications.mobilenet_v2.preprocess_input(window)
                window = np.expand_dims(window, axis=0)

                preds = self.model.predict(window)
                label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

                if label[2] > self.confidence_var.get():
                    box = [int(x / scale), int(y / scale),
                           int((x + 128) / scale), int((y + 128) / scale)]
                    detections.append((box, label[1], label[2]))

        return detections

    def sliding_window(self, image, step_size, window_size):
        for y in range(0, image.shape[0] - window_size[1], step_size):
            for x in range(0, image.shape[1] - window_size[0], step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def update_results(self):
        self.processed_image = self.current_image.copy()
        draw = ImageDraw.Draw(self.processed_image)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Detected Objects:\n\n")

        for (box, label, confidence) in self.detections:
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label}: {confidence:.2f}", fill="red")
            self.results_text.insert(tk.END, f"{label}: {confidence:.2f}\n")

        self.display_image(self.processed_image)
        self.progress.stop()
        self.btn_detect.config(state=tk.NORMAL)

    def update_confidence(self, *args):
        if self.processed_image:
            self.update_results()

    def save_results(self):
        if not self.detections:
            messagebox.showinfo("Info", "No results to save. Please perform object detection first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            results = [{"label": label, "confidence": confidence, "box": box} for (box, label, confidence) in
                       self.detections]
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            messagebox.showinfo("Success", "Results saved successfully")

    def show_preferences(self):
        # Placeholder for preferences dialog
        messagebox.showinfo("Preferences", "Preferences dialog will be implemented here.")

    def toggle_dark_mode(self):
        # Placeholder for dark mode toggle
        messagebox.showinfo("Dark Mode", "Dark mode functionality will be implemented here.")

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.confidence_var.set(config.get('seuil de confiance', 0.5))

    def save_config(self):
        config = {
            'confidence_threshold': self.confidence_var.get()
        }
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


