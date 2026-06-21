# app.py
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg') # Disable GUI window generation for web servers
import matplotlib.pyplot as plt

app = Flask(__name__)

# Setup directories to save calculated graphs
PLOT_DIR = os.path.join('static', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_PATH = "fft_svm_model.pkl"
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

def compute_radial_profile(magnitude_spectrum):
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2).astype(int)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

def analyze_and_generate_plots(image_bytes):
    # Decode Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Process original pipeline
    img_resized = cv2.resize(img_gray, (512, 512))
    filtered_img = cv2.bilateralFilter(img_resized, 7, 60, 60)
    
    # FFT Calculation
    dft = np.fft.fft2(filtered_img)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shift) + 1)

    radial_profile = compute_radial_profile(magnitude_spectrum)
    features = radial_profile[80:220].reshape(1, -1)
    
    # ML Classification
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    confidence = float(probabilities[prediction] * 100)
    
    # Plot 1: Save Input Matrix Image
    plt.figure(figsize=(4,4), facecolor='#0b0f19')
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'input.png'), facecolor='#0b0f19', edgecolor='none')
    plt.close()

    # Plot 2: Save 2D FFT Image
    plt.figure(figsize=(4,4), facecolor='#0b0f19')
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fft.png'), facecolor='#0b0f19', edgecolor='none')
    plt.close()

    # Plot 3: Save 1D Radial Frequency Profile Graph
    fig, ax = plt.subplots(figsize=(4,4), facecolor='#0b0f19')
    ax.plot(radial_profile, color='#00aa66', linewidth=2)
    ax.axvspan(80, 220, color='yellow', alpha=0.1)
    ax.set_facecolor('#0f172a')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'graph.png'), facecolor='#0b0f19', edgecolor='none')
    plt.close()
    
    return int(prediction), round(confidence, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '': return redirect(url_for('home'))
    
    if file:
        image_bytes = file.read()
        prediction, confidence = analyze_and_generate_plots(image_bytes)
        
        if prediction == 1:
            result = "Deepfake Pattern Detected"
        else:
            result = "Authentic Natural Image Verified"
            
        return render_template('result.html', result=result, prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
