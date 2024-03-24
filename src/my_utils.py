'''
    Problem Set 5: Utilities
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains a utility functions. One will create a local copy of the flights database and the 
    will read in a the flights csv 
'''
import pandas as pd
import numpy as np  

######################## Retreive Data ##############################
def get_diamonds_csv() -> None:
    '''  
        Function: get_diamonds_df
        Parameters: None
        Returns: None

        This function will reteive the diamonds.csv from the class repo and create a pandas df
    '''
    path = 'https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv'

    df = pd.read_csv(path)

    return df

######################### Log Scale Vars ##############################
def get_scale_vars(df, feature_var, response_var):
    
    # make copy of explanatory variable column
    X = df[[feature_var]].copy()

    # log scale explanatory variable
    X.carat = np.log(X.carat)
    
    # log scale response variable
    y = np.log(df[response_var])

    return X, y
