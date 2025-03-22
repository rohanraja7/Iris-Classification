# Iris Classification

## Overview
This project implements an Iris classification model using machine learning. The goal is to classify iris flowers into three species: *Setosa*, *Versicolor*, and *Virginica* based on features such as sepal length, sepal width, petal length, and petal width.

## Dataset
The dataset used is the famous **Iris dataset**, which consists of 150 samples (50 for each species). Each sample has the following features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Requirements
To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Files
- `Iris_Classification.ipynb`: Jupyter Notebook containing the data analysis, model training, and evaluation steps.
- `README.md`: This file, providing an overview of the project.

## Steps in the Notebook
1. **Data Loading**: The dataset is loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**: Basic statistics and visualizations to understand the dataset.
3. **Data Preprocessing**: Handling missing values (if any), normalizing data, and encoding labels.
4. **Model Training**: Various machine learning models such as Logistic Regression, Random Forest Classifier and Gradient Boosting Classifiers are trained and evaluated.
5. **Model Evaluation**: Performance metrics such as accuracy, precision, recall, and confusion matrix are computed.
6. **Predictions**: The trained model is used to classify new Iris flower samples.

## Running the Notebook
To execute the project, open the Jupyter Notebook and run the cells step by step:

```bash
jupyter notebook Iris_Classification.ipynb
```

## Results
The best-performing model is identified based on evaluation metrics, and predictions are visualized to understand classification performance.

## Future Improvements
- Implement deep learning models for classification.
- Deploy the model using Flask or FastAPI.
- Enhance feature engineering and hyperparameter tuning.

## Author
This project was developed by Rohan R.

## License
This project is open-source and available under the [MIT License](LICENSE).

