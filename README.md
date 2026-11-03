# MNIST Handwritten Digit Recognition

## Project Overview
This project is a Handwritten Digit Recognition system built using Python, PyTorch, and Convolutional Neural Networks (CNN).

The system can recognize handwritten digits from 0 to 9. It uses the MNIST dataset to train a deep learning model and provides a simple graphical user interface (GUI) where users can draw a digit and get its prediction.

## Technologies Used
- Python
- PyTorch
- CNN (Convolutional Neural Network)
- Pandas
- NumPy
- Matplotlib
- Tkinter
- Pillow (PIL)
- Scikit-learn

## Project Phases

### Phase 1: Data Loading and Preprocessing
- Load the MNIST CSV dataset.
- Separate labels and pixel values.
- Reshape images into 28 × 28 format.
- Normalize pixel values.
- Split data into training and validation sets.

### Phase 2: CNN Model Training
- Build a CNN using PyTorch.
- Use convolution and max-pooling layers.
- Train the model using Adam optimizer.
- Evaluate model performance.
- Save the trained model as `digit_cnn.pth`.

### Phase 3: GUI Development
- Create a drawing interface using Tkinter.
- Allow users to draw digits from 0 to 9.
- Preprocess the drawn image.
- Predict the digit using the trained CNN model.
- Display the predicted digit and confidence score.

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn torch pillow
```

## How to Run

Run the following files in order:

**Step 1: Data Preprocessing**
```bash
python phase1_preprocessing.py
```

**Step 2: Model Training**
```bash
python phase2_model_training.py
```

**Step 3: Launch GUI**
```bash
python phase3_gui.py
```

## Project Files

```text
mnist-task/
│
├── phase1_preprocessing.py
├── phase2_model_training.py
├── phase3_gui.py
├── README.md
├── requirements.txt
├── .gitignore
├── mnist.csv
├── mnist_processed.npz
└── digit_cnn.pth
```

*Note: Dataset and generated model files may be excluded from GitHub using .gitignore.*

## Expected
