import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC

def compute_radial_profile(magnitude_spectrum):
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2).astype(int)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

def extract_fft_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Warning: Image load nahi hui -> {image_path}")
        return None
    
    # 512x512 exact resizing
    img_resized = cv2.resize(img, (512, 512))
    filtered_img = cv2.bilateralFilter(img_resized, 7, 60, 60)
    
    # 2D FFT
    dft = np.fft.fft2(filtered_img)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shift) + 1)
    
    # Extracting the 80-220 frequency band features
    radial_profile = compute_radial_profile(magnitude_spectrum)
    return radial_profile[80:220]

def train_system():
    features = []
    labels = []
    dataset_path = "D:/deepfake detection/FFT+SVM/dataset"
    
    real_dir = os.path.join(dataset_path, "real")
    fake_dir = os.path.join(dataset_path, "fake")
    
    print("====== 🚀 TRAINING PIPELINE INITIALIZED ======")
    
    # Detailed Real Data Loading Log
    real_count = 0
    if os.path.exists(real_dir):
        print(f"📁 Scanning Real Images Folder: {real_dir}")
        for file in os.listdir(real_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                feat = extract_fft_features(os.path.join(real_dir, file))
                if feat is not None:
                    features.append(feat)
                    labels.append(0) # 0 for Real
                    real_count += 1
        print(f"✅ Real Images Processed: {real_count}")
    else:
        print(f"🚩 Error: Real directory nahi mili at {real_dir}")

    # Detailed Fake Data Loading Log
    fake_count = 0
    if os.path.exists(fake_dir):
        print(f"📁 Scanning Fake Images Folder: {fake_dir}")
        for file in os.listdir(fake_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                feat = extract_fft_features(os.path.join(fake_dir, file))
                if feat is not None:
                    features.append(feat)
                    labels.append(1) # 1 for Fake
                    fake_count += 1
        print(f"✅ Fake Images Processed: {fake_count}")
    else:
        print(f"🚩 Error: Fake directory nahi mili at {fake_dir}")

    print("---------------------------------------------")
    print(f"📊 Total Dataset Loaded successfully: {len(features)} images")
    
    if len(features) == 0:
        print("❌ Error: No features extracted. Training aborted!")
        return

    X = np.array(features)
    y = np.array(labels)

    print("🧠 Training Support Vector Machine (SVM) Classifier...")
    # Exact old configuration
    clf = SVC(kernel='rbf', C=20.0, gamma='scale', class_weight='balanced', probability=True)
    clf.fit(X, y)

    # Saving the model brain
    model_output_path = "D:/deepfake detection/FFT+SVM/fft_svm_model.pkl"
    with open(model_output_path, "wb") as f:
        pickle.dump(clf, f)
    
    print("---------------------------------------------")
    print(f"🏆 SUCCESS: Model saved at -> {model_output_path}")
    print("====== 🎉 PIPELINE COMPLETED PERFECTLY ======")

if __name__ == "__main__":
    train_system()