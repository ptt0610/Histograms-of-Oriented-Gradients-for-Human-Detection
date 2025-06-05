import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage import color
from imutils.object_detection import non_max_suppression


# Define constants
IMG_SIZE = (64, 128)  # (height, width) for training and feature extraction
DETECTION_SIZE = (64, 128)  # Larger size for detection
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# Load the model and scaler
model = joblib.load('Others/hog_svm_model.npy')
scaler = joblib.load('Others/hog_scaler.npy')

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def detect_pedestrians(image, model, scaler, winW=64, winH=128, stepSize=10, threshold=0.2, orientations=9):
    detections = []
    hog_image = None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Resize the entire image to match training size
    image_resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    if image_resized.shape != (IMG_SIZE[1], IMG_SIZE[0]):  # (height, width)
        raise ValueError(f"Resized image has size {image_resized.shape}, expected ({IMG_SIZE[1]}, {IMG_SIZE[0]})")

    # Compute HOG visualization for the entire image
    _, hog_image = hog(image_resized, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                       cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=True)
    hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    total_windows = 0
    positive_predictions = 0
    high_confidence = 0

    # Apply sliding window on the resized image
    for (x, y, window) in sliding_window(image_resized, stepSize=stepSize, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        total_windows += 1
        
        window_resized = window
        if np.max(window_resized) == 0:
            print(f"Warning: Window at ({x}, {y}) is black")
            continue
        
        fds = hog(window_resized, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                  cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
        fds = fds.reshape(1, -1)
        fds_scaled = scaler.transform(fds)
        pred = model.predict(fds_scaled)
        score = model.decision_function(fds_scaled)[0]
        
        if pred == 1:
            positive_predictions += 1
            print(f"Positive prediction at ({x}, {y}), Score: {score:.4f}")
            detections.append((x, y, score, winW, winH))
            if score > threshold:
                high_confidence += 1
                print(f"Detection:: Location -> ({x}, {y}), Score -> {score:.4f}")

    print(f"Total windows checked: {total_windows}")
    if total_windows > 0:
        print(f"Positive predictions: {positive_predictions} ({positive_predictions/total_windows*100:.2f}%)")
        print(f"High confidence detections: {high_confidence} ({high_confidence/total_windows*100:.2f}%)")
    else:
        print("No valid windows found.")
    return detections, total_windows, positive_predictions, hog_image


# GUI Application
class PedestrianDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedestrian Detector")
        self.root.geometry("1500x600")  # Maintain current size
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.image_path = tk.StringVar()
        self.image_tk = None
        self.detected_image_tk = None
        self.hog_image_tk = None

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Main container frames for left, center, and right sides
        left_frame = tk.Frame(self.root, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        center_frame = tk.Frame(self.root, bg="#f0f0f0")
        center_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(self.root, bg="#f0f0f0")
        right_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Left Side: Load Image Button and Input Image
        tk.Button(
            left_frame, 
            text="Load Image", 
            command=self.load_image, 
            bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
            relief=tk.RAISED, padx=10, pady=5
        ).pack(pady=10)

        tk.Label(left_frame, text="Input Image", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=5)
        self.image_frame = tk.Label(left_frame, bg="#ffffff", relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Center Side: Detect Button and Detected Image
        self.detect_button = tk.Button(
            center_frame, 
            text="Detect Pedestrians", 
            command=self.detect_pedestrians_gui, 
            state=tk.DISABLED,
            bg="#2196F3", fg="white", font=("Arial", 12, "bold"),
            relief=tk.RAISED, padx=10, pady=5
        )
        self.detect_button.pack(pady=10)

        tk.Label(center_frame, text="Detected Image", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=5)
        self.detected_frame = tk.Label(center_frame, bg="#ffffff", relief=tk.SUNKEN, borderwidth=2)
        self.detected_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Right Side: Placeholder for imaginary button and HOG Visualization
        tk.Button(
            right_frame, 
            text="",  # Empty text for imaginary button placeholder
            state=tk.DISABLED,  # Disabled to prevent interaction
            bg="#f0f0f0", fg="#f0f0f0",  # Match background to hide it
            font=("Arial", 12, "bold"),
            relief=tk.FLAT, padx=10, pady=5
        ).pack(pady=10)  # Placeholder for imaginary button

        tk.Label(right_frame, text="HOG Visualization", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=5)
        self.hog_frame = tk.Label(right_frame, bg="#ffffff", relief=tk.SUNKEN, borderwidth=2)
        self.hog_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path.set(file_path)
            image = Image.open(file_path).convert('L')
            image = image.resize((300, 400), Image.Resampling.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(image)
            self.image_frame.config(image=self.image_tk)
            self.image_frame.image = self.image_tk
            self.root.update_idletasks()
            self.detect_button.config(state=tk.NORMAL)

    def detect_pedestrians_gui(self):
        if not self.image_path.get():
            messagebox.showerror("Error", "Please load an image first!")
            return

        image = cv2.imread(self.image_path.get(), cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Error", "Failed to load the image!")
            return

        # Detect pedestrians and get HOG visualization with fixed orientations=9
        detections, total_windows, positive_predictions, hog_image = detect_pedestrians(
            image, model, scaler, threshold=0.2, orientations=9
        )

        # Apply non-maximum suppression
        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) if detections else np.array([])
        scores = np.array([score for (_, _, score, _, _) in detections]) if detections else np.array([])
        pick = non_max_suppression(rects, probs=scores, overlapThresh=0.3) if len(rects) > 0 else []

        # Prepare original image for visualization
        img_display = image.copy()
        if np.max(img_display) > np.min(img_display):
            img_display = (img_display - np.min(img_display)) / (np.max(img_display) - np.min(img_display)) * 255
        img_display = img_display.astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

        # Draw rectangles on detected image
        detected_img = img_display.copy()
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(detected_img, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # Prepare HOG image
        hog_image = cv2.resize(hog_image, (300, 400), interpolation=cv2.INTER_AREA)
        hog_image = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2RGB)

        # Display images
        detected_image = Image.fromarray(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
        detected_image = detected_image.resize((300, 400), Image.Resampling.LANCZOS)
        self.detected_image_tk = ImageTk.PhotoImage(detected_image)
        self.detected_frame.config(image=self.detected_image_tk)
        self.detected_frame.image = self.detected_image_tk

        hog_image_pil = Image.fromarray(hog_image)
        self.hog_image_tk = ImageTk.PhotoImage(hog_image_pil)
        self.hog_frame.config(image=self.hog_image_tk)
        self.hog_frame.image = self.hog_image_tk

        messagebox.showinfo("Detection Result", f"Total windows checked: {total_windows}\nPositive predictions: {positive_predictions}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PedestrianDetectorGUI(root)
    root.mainloop()