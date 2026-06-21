import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pickle

def compute_radial_profile(magnitude_spectrum):
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2).astype(int)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Original Hybrid FFT-SVM Analyzer Dashboard")
        self.root.geometry("1150x680")
        self.root.configure(bg="#121212")

        # Exact model file path loading
        self.model_path = "D:/deepfake detection/FFT+SVM/fft_svm_model.pkl"
        
        print("====== 🔍 LAUNCHING INFERENCE ENGINE ======")
        if os.path.exists(self.model_path):
            print(f"📁 Loading trained SVM model brain from: {self.model_path}")
            with open(self.model_path, "rb") as f:
                self.clf = pickle.load(f)
            self.system_status = "🤖 Model Loaded Successfully: Original FFT Inference Ready"
            self.status_color = "#00ff88"
            print("✅ Model successfully initialized in memory.")
        else:
            self.clf = None
            self.system_status = "⚠️ Warning: 'fft_svm_model.pkl' nahi mila! Pehle trainer.py chalayein."
            self.status_color = "#ffaa00"
            print("🚩 Warning: Active model file not found on disk.")

        # GUI Interface Elements
        self.title_label = tk.Label(root, text="HYBRID SIGNAL PROCESSING & MACHINE LEARNING DASHBOARD", font=("Helvetica", 15, "bold"), fg="#ffffff", bg="#1f1f1f", pady=12)
        self.title_label.pack(fill=tk.X)
        
        self.control_frame = tk.Frame(root, bg="#121212", pady=10)
        self.control_frame.pack()
        
        self.btn_select = tk.Button(self.control_frame, text="📁 Browse & Scan Image", font=("Helvetica", 11, "bold"), bg="#00aa66", fg="white", padx=20, pady=8, relief=tk.FLAT, command=self.process_image)
        self.btn_select.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(root, text=self.system_status, font=("Helvetica", 10), fg=self.status_color, bg="#121212")
        self.status_label.pack()
        
        self.result_label = tk.Label(root, text="System Live: Ready to scan target image signal matrix...", font=("Helvetica", 14, "bold"), fg="#888888", bg="#121212", wraplength=950)
        self.result_label.pack(pady=10)
        
        self.plot_frame = tk.Frame(root, bg="#121212")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas = None

    def process_image(self):
        if self.clf is None:
            messagebox.showerror("Error", "Trained ML model (.pkl file) available nahi hai! Pehle system ko train karein.")
            return

        target_folder = "D:/deepfake detection/FFT+SVM"
        if not os.path.exists(target_folder):
            target_folder = os.getcwd()

        file_path = filedialog.askopenfilename(initialdir=target_folder, title="Select Target Image Matrix", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        print(f"\n⚡ Scanning target signal: {file_path}")

        # 1. Image Processing Pipeline
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (512, 512))
        filtered_img = cv2.bilateralFilter(img_resized, 7, 60, 60)
        
        # 2D Fast Fourier Transform
        dft = np.fft.fft2(filtered_img)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(np.abs(dft_shift) + 1)

        # Feature Vector Formulation
        radial_profile = compute_radial_profile(magnitude_spectrum)
        test_feature = radial_profile[80:220].reshape(1, -1)

        # 2. Machine Learning Classification
        prediction = self.clf.predict(test_feature)[0]
        probabilities = self.clf.predict_proba(test_feature)[0]
        confidence = probabilities[prediction] * 100

        print(f"📊 Raw Model Signatures -> Class 0 (Real): {probabilities[0]*100:.2f}%, Class 1 (Fake): {probabilities[1]*100:.2f}%")

        if prediction == 1:
            message = f"🚩 WARNING: DEEPFAKE / AI GENERATED DETECTED!\n(ML Confidence Verdict: {confidence:.2f}%)"
            self.result_label.config(text=message, fg="#ff4444")
            print("🚨 Verdict: DEEPFAKE DETECTED")
            messagebox.showwarning("ML Verdict", "Statistical anomalies found in spectrum frequency band. Image is AI Generated.")
        else:
            message = f"✅ SAFE: REAL ORIGINAL IMAGE VERIFIED\n(ML Confidence Verdict: {confidence:.2f}%)"
            self.result_label.config(text=message, fg="#00ff88")
            print("🍏 Verdict: REAL ORIGINAL IMAGE")
            messagebox.showinfo("ML Verdict", "Frequency behaviors match natural physical laws. Image is Real.")

        # 3. Dynamic Visualizations Update
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), facecolor='#121212')
        
        cv_img = cv2.imread(file_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ax1.imshow(cv_img)
        ax1.set_title('Target Matrix input', color='white', fontsize=10)
        ax1.axis('off')
        
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.set_title('2D Fourier Power Spectrum', color='white', fontsize=10)
        ax2.axis('off')
        
        ax3.plot(radial_profile, color='#00aa66', linewidth=2)
        ax3.axvspan(80, 220, color='yellow', alpha=0.1)
        ax3.set_title('1D Power Spectrum Distribution', color='white', fontsize=10)
        ax3.tick_params(colors='white', labelsize=8)
        ax3.set_facecolor('#1e1e1e')
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    root.mainloop()