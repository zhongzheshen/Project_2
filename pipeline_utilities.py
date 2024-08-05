from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.svm import SVR



#select columns to drop based on data exploration
def drop_columns(df):
    df = df.drop(columns=['Event','Gender','Year']) 
    return df

def preprocess_olympics_data(df):
    """
    Written for Summer olympics data; will split into training
    and testing sets. Uses Medal as the target column.
    """
    X = df.drop(columns='Medal')
    y = df['Medal'].values.reshape(-1, 1)
    return train_test_split(X, y)

def r2_adj(x, y, pipeline):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = pipeline.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, pipeline):
    # Use the pipeline to make predictions
    y_pred = pipeline.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, pipeline)}")
    return r2_adj(X_test, y_test, pipeline)




def get_best_pipeline(pipeline1, pipeline2,pipeline3,pipeline4,df):
    """
    Accepts two pipelines and olympics medals data.
    Uses two different preprocessing functions to 
    split the data for training the different 
    pipelines, then evaluates which pipeline performs
    best.
    """
    # Apply the preprocess_rent_data step
    X_train, X_test, y_train, y_test = preprocess_olympics_data(df)

    # Fit the first pipeline
    pipeline1.fit(X_train, y_train)
    print("Testing Linear Regression")

    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the first pipeline
    p1_adj_r2 = check_metrics(X_test, y_test, pipeline1)
    print("------------------------------------------")
    
    # Fit the second pipeline
    pipeline2.fit(X_train, y_train)
    print("Testing Random Forest Regressor")
    
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    p2_adj_r2 = check_metrics(X_test, y_test, pipeline2)
    print("------------------------------------------")
    
    # Fit the third pipeline
    pipeline3.fit(X_train, y_train)
    print("Testing XGB Regressor")
    
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    p3_adj_r2 = check_metrics(X_test, y_test, pipeline3)
    print("------------------------------------------")
    
    # Fit the fourth pipeline
    pipeline4.fit(X_train, y_train)
    print("Testing SVR Regressor")
    
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    p4_adj_r2 = check_metrics(X_test, y_test, pipeline4)
    print("------------------------------------------")
    
    # Compare the adjusted r-squared for each pipeline and 
    # return the best model
    if p1_adj_r2 > p2_adj_r2:
        if p1_adj_r2 > p3_adj_r2:
            if p1_adj_r2 > p4_adj_r2:
                print("Linear Regression is the best model.")
                return pipeline1
            else:
                print("SVR Regressor is the best model.")
                return pipeline4
        elif p3_adj_r2 > p4_adj_r2:
            print("XGB Regressor is the best model.")
            return pipeline3
    elif p2_adj_r2 > p3_adj_r2:
        if p2_adj_r2 > p4_adj_r2:
            print(("Random Forest Regressor is the best model."))
            return pipeline2
        else:
            print("SVR Regressor is the best model.")
            return pipeline4
    elif p3_adj_r2 > p4_adj_r2:
        print("XGB Regressor is the best model.")
        return pipeline3
    else: 
        print("SVR Regressor is the best model.")
        return pipeline4
    

#select models to test
def medal_model_generator(df):
    """
    Defines a series of steps that will preprocess data,
    split data, and train a model for predicting rent prices
    using linear regression. It will return the best trained model
    and print the mean squared error, r-squared, and adjusted
    r-squared scores.
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function

    model1 = [("Linear Regression", LinearRegression())]

    model2 = [("Random Forest Regressor", RandomForestRegressor())] 

    model3 = [("XGB Regressor", XGBRegressor())] 

    model4 = [('Linear SVR', SVR())]



    # Create a pipeline object
    pipeline1 = Pipeline(model1)

    # Create a second pipeline object
    pipeline2 = Pipeline(model2)

    # Create a third pipeline object
    pipeline3 = Pipeline(model3)

    # Create a fourth pipeline object
    pipeline4 = Pipeline(model4)

    # Get the best pipeline
    pipeline = get_best_pipeline(pipeline1, pipeline2, pipeline3,pipeline4,df)


    # Return the trained model
    return pipeline



if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")