# [PROJECT 2 - GROUP 1]

## EXECUTIVE SUMMARY: Overview and Objectives

The Summer Olympic Games represent the pinnacle of international sports competition, where athletes from around the world compete for glory and national pride. Our goal with this project is to develop predictive models that can estimate the 2016-2024 medal counts for the United States of America (USA) based on past data (1896-2012). By leveraging machine learning algorithms, specifically **Linear Regression**, **Random Forest Regressor**, **XGBoost Regressor**, and **SVR**. , we aim to achieve precise predictions.



## OVERVIEW OF THE DATA COLLECTION, CLEAN UP AND EXPLORATION PROCESS 

* The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/the-guardian/olympic-games?select=summer.csv)

### Preprocessing Steps

#### Data Cleaning
- Handling missing values by imputing relevant statistics (e.g., mean or median) for numerical columns.
- Dropping non-relevant columns such as 'Athlete', 'Sport', and 'Discipline'.

#### Data Encoding
- Encoding categorical variables using One Hot Encoding for features like event type and athlete nationality.

#### Data Splitting
- Splitting the dataset into training and testing sets using the `train_test_split` function.

#### Exploratory Data Analysis
- Aggregated medal counts by country, year, and type.
- Identified and removed outliers due to historical events, such as the 1904 and 1984 Olympics.

#### Data Visualization
- Plotted USA medal counts over the years.
- Computed and visualized the correlation between year and medal counts.


## Model Training and Evaluation

#### Pipeline for Model Training
A custom pipeline was created to preprocess data, split the dataset, and train various models. The models used include Linear Regression, Random Forest Regressor, XGBoost Regressor, and SVR. The pipeline includes the following steps:

#### Data Preprocessing:
- Drop non-relevant columns.
- Split the dataset into features (X) and target (y).
Model Training:

#### Create pipelines for each model.
- Fit each model on the training data.
- Evaluate each model using Mean Squared Error (MSE), R-squared (R²), and Adjusted R-squared.

#### Model Evaluation:
- Calculate and compare the performance metrics for each model.
- Select the best-performing model based on adjusted R-squared values.


## Observations and Results

### Linear Regression
- Provided a solid baseline with reasonable R-squared values.

### Random Forest Regressor
- Showed improved performance with a lower Mean Squared Error.

### XGBoost Regressor
- Achieved the highest accuracy among the evaluated models with the best R-squared value.

### SVR Regressor
- Also performed well, with comparable results to the other models.

Each model was evaluated using metrics such as Mean Squared Error (MSE), R-squared (R²), and Adjusted R-squared.


## Observations and Results

### Linear Regression
- Provided a solid baseline with reasonable R-squared values.

### Random Forest Regressor
- Showed improved performance with a lower Mean Squared Error.

### XGBoost Regressor
- Achieved the highest accuracy among the evaluated models with the best R-squared value.


## Results and Conclusions
### Testing Linear Regression
- Mean Squared Error: 169.13767243728768
- R-squared: 0.9812702073506133
- Adjusted R-squared: 0.9625404147012266

### Testing Random Forest Regressor
- Mean Squared Error: 668.1121999999997
- R-squared: 0.926015282153269
- Adjusted R-squared: 0.852030564306538

### Testing XGB Regressor
- Mean Squared Error: 1577.9070005943565
- R-squared: 0.8252673745155334
- Adjusted R-squared: 0.6505347490310669

### Testing SVR Regressor
- Mean Squared Error: 9317.099648585581
- R-squared: -0.031747345207108424
- Adjusted R-squared: -1.0634946904142168
  
### Linear Regression is the best model.
