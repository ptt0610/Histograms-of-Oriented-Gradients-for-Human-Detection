HOG + SVM Pedestrian Detector Project
=====================================

This project implements a pedestrian detection system using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier. It includes an ablation study to evaluate HOG orientation bins and a GUI for visualizing detection results [1].

**Note**
The training data images are supposed to be put in the "Others" folder due to the project's requirements. Because the data size is too large to upload to Github, the Other folders will only contain the saved model after training and plot results. Future users can put training data into 2 folders, "negative images" and "positive images" inside the "Others" folder to re-run the program.

Data Source:
The positive images are taken from the PETA image source [2]. All positive images are in size 64 width *128 height. The negative images are taken from the Kaggle Source [3], which consists of 19558 images of street background photos. There are approximately 120 images, including the pedestrians on the street and the pedestrians are very small compared to the background. These photos are not good for the purpose of training because they can make noise with pedestrian edges inside the negative images, so after selecting 2000 images, I manually check to ensure the negative images used for training and testing do not include any pedestrians. All negative images are in size 224x224, they will be cropped and resized to size 64x128 in the preprocessing phase.

<img width="278" alt="image" src="https://github.com/user-attachments/assets/f66d0001-3776-4dbb-802c-adc3e2aa6148" />


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


**RESULT**


<img width="329" alt="image" src="https://github.com/user-attachments/assets/63288e02-5396-4262-92b7-5662c198d4ae" />


<img width="337" alt="image" src="https://github.com/user-attachments/assets/db1847ef-338f-4f70-8567-b4f2ef562d48" />


Testing with GUI

<img width="956" alt="image" src="https://github.com/user-attachments/assets/c4152a76-1f57-43ac-8b7f-3b981fc42786" />

<img width="957" alt="image" src="https://github.com/user-attachments/assets/4c120065-f1e8-4f22-809b-06c3ed19ef63" />


## RREFERENCE ##

[1] Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
CVPR 2005.

[2] Deng, Y., Luo, P., Loy, C. C., & Tang, X. (2014). Pedestrian attribute recognition at far distance. In Proceedings of the 22nd ACM international conference on Multimedia (pp. 789–792). ACM. https://mmlab.ie.cuhk.edu.hk/projects/PETA.html

[3] Mike Mazurov. House Rooms & Streets Image Dataset. https://www.kaggle.com/datasets/mikhailma/house-rooms-streets-image-dataset?resource=download






