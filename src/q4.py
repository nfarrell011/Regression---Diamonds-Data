'''
    Problem Set 5: q4
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023
'''
from my_utils import get_diamonds_csv
from my_utils import get_scale_vars
import numpy as np  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

#################################### Find Best Model Function ##########################################
def find_best_model(df: pd.DataFrame, col: pd.Series) -> float and pd.DataFrame :
    '''
        Function: find_best_model
        Parameters: 1 list
            col: the column catigorical column to encode and use in the model
        Returns: 1 float, 1 pd.Frame
            r_squared: the model score
            X: the feature matrix

        This function will execute an MLR. It will encode the arguement column and 
            return the coefficent of determination as the model score.
    '''
    # extract and scale feature matrix and response var
    X, y = get_scale_vars(df, 'carat', 'price')

    # select the catagorical variable to encode
    column_to_encode = df[[col]]

    # instantiate encoder
    encoder = OneHotEncoder(drop = 'first')

    # encode catagorical variable
    enc_data = pd.DataFrame(encoder.fit_transform(column_to_encode).toarray()) 

    # concat encoded variable to feature matrix
    X = pd.concat([X, enc_data], axis = 1)

    # change col names to strings
    X.columns = X.columns.astype(str)

    # instaniate model
    reg = LinearRegression(fit_intercept = True)

    # fit model and get r squared
    reg.fit(X, y)
    r_squared = reg.score(X, y)

    # return r squared and feature matrix
    return r_squared, X

################################################################################################
########################################### Main ###############################################
################################################################################################

def main():

    df = get_diamonds_csv()

    # these are the catagorical variables to test
    cols = ['cut', 'color', 'clarity']

    # this will invoke find_best_model on each col, and return 2 tuples
    r_squared_list, feature_matrix_list = zip(*[find_best_model(df, i) for i in cols])

    # capitalize col names for plot purposes
    cols = [col.capitalize() for col in cols]

    # plot results
    sns.scatterplot(x = cols, y = r_squared_list, marker = '*', s = 200, color = 'firebrick')
    plt.title(f'Model Score \n with Inclusion of Catagorical Variable', weight = 'bold')
    plt.xlabel('Variable Included in Model', weight = 'bold')
    plt.ylabel('$R^2$')
    
    # save figure
    save_path = ('figs/best_model_fig_1.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close();

    # find the max of r_squared_list index, to extract the best model
    max_index = np.argmax(r_squared_list)

    # extract best model
    best_variable = cols[max_index]
    best_r_squared = r_squared_list[max_index]
    best_feature_matrix_shape = feature_matrix_list[max_index].shape

    # print results
    print(f'The additional variable that performed the best is: {best_variable}.')
    print(f'Using {best_variable} in the model, the coefficient of determination is: {best_r_squared:.4f}.')
    print(f'The shape of the feature matrix for this model is: {best_feature_matrix_shape}.')

if __name__ == "__main__":
    main()