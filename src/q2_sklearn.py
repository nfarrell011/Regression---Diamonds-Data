'''
    Problem Set 5: q2
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file will execute an SLR using sklearn with price and carat both log scaled
'''
from my_utils import get_diamonds_csv
from my_utils import get_scale_vars
from sklearn.linear_model import LinearRegression
import pandas as pd

###################### Stats Models: SLR #############################
def sklearn_SLR(features: pd.DataFrame, response: pd.DataFrame):
    '''
        Function: stats_model_SLR
        Parameters: 2 pd.DataFrames
            features: the design matrix
            reponse: the reponse variable
        Returns: 1 float, 1 statsmodels.regression.linear_model.RegressionResultsWrapper
            r_squared: the coefficient of determination of the linear model

        This function will fit an sklearn linear regression
    '''
    reg = LinearRegression(fit_intercept = True)

    reg.fit(features, response)

    sklearn_r_squared = reg.score(features, response)

    return sklearn_r_squared

################################# Main #############################
def main():
    
    # get data: this function is defined in my_utils
    df = get_diamonds_csv()

    # log_scale both variables
    X, y = get_scale_vars(df, 'carat', 'price')

    # execute model
    sklearn_r_2 = sklearn_SLR(X, y)

    # display results
    print(f'The coefficient of determination for the sklearn linear model is {sklearn_r_2:.4f}.')

if __name__ == "__main__":
    main()