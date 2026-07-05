\# Iris Flower Classification



\## Task 1 - Oasis Infobyte Internship



\## Objective

The objective of this project is to build a machine learning classification model to identify the species of an Iris flower as Setosa, Versicolor, or Virginica based on its physical measurements.



\## Dataset

The Iris dataset is loaded directly from `sklearn.datasets.load\_iris()`. No external dataset download is required.



\## Features Used

\- Sepal length

\- Sepal width

\- Petal length

\- Petal width



\## Technologies Used

\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- Joblib



\## Project Workflow

1\. Loaded the Iris dataset.

2\. Performed Exploratory Data Analysis.

3\. Checked shape, data types, null values, and descriptive statistics.

4\. Created visualizations such as pairplot, boxplots, heatmap, and confusion matrices.

5\. Split the dataset into training and testing sets.

6\. Trained four classification models:

&#x20;  - Logistic Regression

&#x20;  - K-Nearest Neighbors

&#x20;  - Decision Tree

&#x20;  - Random Forest

7\. Evaluated each model using accuracy, confusion matrix, and classification report.

8\. Selected the best-performing model.

9\. Saved the best model using Joblib.



\## Best Model

The best-performing model was \*\*K-Nearest Neighbors\*\*, which achieved \*\*100% accuracy\*\* on the test data.



\## Feature Selection Discussion

Petal length and petal width are the most discriminative features because they show clear separation between the three Iris species, especially Setosa compared to Versicolor and Virginica.



\## How to Run

1\. Install the required libraries:



```bash

pip install -r requirements.txt

