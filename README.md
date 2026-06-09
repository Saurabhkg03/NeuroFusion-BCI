# NeuroFusion-BCI: Advanced Motor Imagery Classification 🧠⚡

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository contains the implementation of a high-performance Brain-Computer Interface (BCI) classification system. The project focuses on decoding 4-class motor imagery tasks (Left Hand, Right Hand, Foot, and Tongue) from raw EEG signals. 

By proposing a novel **Hybrid Feature Fusion** methodology that bridges traditional time-domain features with non-linear complexity measures, this model achieves highly robust spatial and temporal pattern recognition using a custom BiGRU + 1D CNN deep learning architecture.

## ✨ Key Contributions
* **Novel Feature Engineering:** Combined baseline Hjorth Parameters (Activity, Mobility, Complexity) with advanced signal complexity metrics including Petrosian Fractal Dimensions (PFD) and Spectral Entropy.
* **Hybrid Architecture:** Engineered a sequential deep learning pipeline utilizing Bidirectional GRUs (BiGRU) to capture long-term temporal dependencies, followed by 1D Convolutional Neural Networks (CNN) for local spatial feature extraction.
* **Overfitting Prevention:** Integrated Spatial Dropout and L2 Regularization techniques to ensure high generalization on unseen subject data.
* **Automated Evaluation:** Includes a full pipeline for generating publication-ready comparative visualizations, confusion matrices, and performance metric tables.

## 📊 Results & Performance
The proposed Hybrid Fusion model significantly outperforms baseline approaches. 

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | `96.88%` |
| **Macro F1-Score** | `0.9686` |
| **Cohen's Kappa** | `0.9583` |

*(Note: You can add the generated bar charts and confusion matrix images in the `results/` folder and link them here).*

## 🏗️ System Architecture

1. **Preprocessing:** Artifact removal and bandpass filtering of raw EEG multi-channel arrays.
2. **Feature Extraction Block:** - Time-domain: Hjorth Parameters
   - Non-linear/Frequency-domain: Petrosian Fractal Dimension, Spectral Entropy
3. **Deep Learning Block:**
   - `BiGRU Layers` (captures bidirectional temporal sequences)
   - `Conv1D Layers` (extracts localized spatial patterns)
   - `GlobalAveragePooling1D` & `MaxPooling1D`
   - `Dense Layers` with Softmax activation for 4-class output

## 💻 Installation & Setup

1. **Clone the repository:**
   
```bash
   git clone [https://github.com/yourusername/NeuroFusion-BCI.git](https://github.com/yourusername/NeuroFusion-BCI.git)
   cd NeuroFusion-BCI
   

```

2. **Create a virtual environment (optional but recommended):**

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   

```

3. **Install dependencies:**

```bash
   pip install -r requirements.txt
   

```

## 🚀 Usage

**1. Data Preparation:**
Place your preprocessed `.npy` or `.csv` EEG data files into the `data/` directory. Ensure the data shape aligns with `(samples, time_steps, channels)`.

**2. Training the Model:**
Run the Jupyter Notebook `Final_EEG_Project.ipynb` or execute the training script:

```bash
python train.py --epochs 50 --batch_size 32

```

**3. Evaluation:**
To generate performance metrics and visualizations on the test set:

```bash
python evaluate.py --model_path models/best_hybrid_model.h5

```

## 📂 Project Structure

```text
├── data/                   # Raw and processed dataset files (ignored in git)
├── models/                 # Saved model weights (.h5)
├── results/                # Output images, confusion matrices, and tables
├── src/                    # Source code for modular use
│   ├── extract_features.py # Hjorth, PFD, and Entropy logic
│   ├── model.py            # BiGRU + CNN architecture definition
│   └── utils.py            # Plotting and evaluation helpers
├── Final_EEG_Project.ipynb # Main exploratory and training notebook
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```

## 👨‍💻 Author

**Saurabh Gaikwad**

* Portfolio: [Insert your website link here]
* LinkedIn: [Insert LinkedIn link]
* GitHub: [@yourusername](https://github.com/yourusername)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.


