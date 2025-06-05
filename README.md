HOG + SVM Pedestrian Detector Project
=====================================

This project implements a pedestrian detection system using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier. It includes an ablation study to evaluate HOG orientation bins and a GUI for visualizing detection results.

Prerequisites
-------------
1. **Python Environment**:
   - Ensure Python 3.7 or later is installed.

2. **Install Dependencies**:
   - A `requirements.txt` file lists all required packages.
   - Run the provided `install_requirements.py` script to install dependencies automatically
     This script installs packages like `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `tqdm`, `joblib`, `opencv-python`, and `pillow`.

3. **Dataset**:
- Prepare training and test datasets:
- Training: 4000 grayscale images (e.g., in `train_images/` directory).
- Test: 200 grayscale images (e.g., in `test_images/` directory).
- Labels should indicate pedestrian (1) or non-pedestrian (0).

Step 1: Run notebook for the Detector and the Ablation Study
------------------------------
1. Open `HOG_SVM_pedestrian_detector.ipynb` in Jupyter Notebook
2. Update the notebook to load your dataset:
- Modify the data loading section to match your image paths
3. Run all cells in the notebook


Step 2: Launch the GUI
----------------------
1. Ensure `GUI.py` is in the same directory as `hog_svm_model.npy` and `hog_scaler.npy`.
2. Run the GUI script with the command: ``python GUI.py``
- Click "Load Image" to select a test image (.jpg, .jpeg, .png).
- Click "Detect Pedestrians" to run detection and view results:
  - Left: Original image.
  - Center: Detected image with green rectangles around pedestrians.
  - Right: HOG visualization.
(The GUI uses the 9-bin model by default (orientations=9).

Step 3: Testing images using GUI
----------------------
Run the GUI with images in Testing Images folder and save prediction result to prediction.xlsx

