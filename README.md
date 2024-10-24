

# Diabetes Detection using ML


## Overview
This project focuses on predicting diabetes using machine learning models. The dataset contains various health indicators such as glucose levels, blood pressure, insulin levels, BMI, and more. The goal is to build a machine learning model that accurately predicts whether a person has diabetes based on these features.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Modeling Approach](#modeling-approach)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```
├── data/
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv  # Main dataset
├── notebooks/
│   ├── data_preprocessing.ipynb                         # Data cleaning and preprocessing
│   ├── model_training.ipynb                             # Model building and training
├── README.md                                            # Project documentation
└── scripts/
    ├── train_model.py                                   # Python script to train the model
    ├── evaluate_model.py                                # Script for model evaluation
```

## Dataset
The dataset used in this project contains the following features:
- **Pregnancies**
- **Glucose**
- **Blood Pressure**
- **Skin Thickness**
- **Insulin**
- **BMI (Body Mass Index)**
- **Diabetes Pedigree Function**
- **Age**
- **Outcome** (0 or 1, indicating whether the patient has diabetes)

You can find the dataset [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database) or similar diabetes health indicator datasets.

## Dependencies
To run the code, you will need the following libraries:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook 
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Modeling Approach
### Data Preprocessing
- Handled missing values
- Normalized numerical features
- Split data into training and testing sets

### Model Selection
We experimented with multiple machine learning algorithms, including:
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Model Evaluation
The models were evaluated based on accuracy, precision, recall, and F1-score using the test dataset.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd diabetes-detection
   ```
3. Run the data preprocessing notebook or script:
   ```bash
   jupyter notebook notebooks/data_preprocessing.ipynb
   ```
4. Train the model using:
   ```bash
   python scripts/train_model.py
   ```
5. Evaluate the model:
   ```bash
   python scripts/evaluate_model.py
   ```

