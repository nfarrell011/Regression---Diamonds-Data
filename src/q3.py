'''
    Problem Set 5: q3
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023
'''
from my_utils import get_diamonds_csv
from my_utils import get_scale_vars

import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer, r2_score

##################################### Bogden's Train Test Split ##############################################
def simple_train_test_split(X, y, test_size=.3):
    '''  
        Function: simple_test_train_split
        Parameters: 1 pd.DataFrame, 1 pd.Series, 1 float
        Returns: 2 pd.DataFrames, 2 pd.Series

        This function split a feature matrix and a reponse variable in train and test groups
    '''
    n_training_samples = int((1.0 - test_size) * X.shape[0])

    X_train = X[:n_training_samples,:]
    y_train = y[:n_training_samples]

    X_test = X[n_training_samples:,:]
    y_test = y[n_training_samples:]

    return X_train, X_test, y_train, y_test

######################################## SLR Various Test Sizes ################################################
def slr_var_test_size(X, y, test_sizes: list = [.3], sklearn: bool  = True) -> list:
    ''' 
        Function: slr_var_test_size

        Parameters: 2 pd.Series, 1 list, 1 bool
            X: feature
            y: target
            test_sizes: list of test proportions to use in the model
            sklearn: a bool determining with train test split function to use

        Returns: 1 array
            r_squared_array: an array containg the coeffecient of determinations for the SLRs with 
            varying train/test sizes

        This function will execute multiple SLRs using sklearn with different training and test sizes and return an 
            array of the coefficient of determinations for different models.
    '''
    # initialize r_squared list
    r_squared_array = []
    
    # iterate over the test sizes
    for i in test_sizes:

        # pick train test split function
        if sklearn == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= i)
        else:
            X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size= i)
        
        # execute model
        reg = LinearRegression(fit_intercept = True)
        reg.fit(X_train, y_train)
        y_hat = reg.predict(X_test)
        r_squared = r2_score(y_test, y_hat)

        # update return array
        r_squared_array.append(r_squared)

    return r_squared_array

############################################# R^2 Graphic ##################################################
def comparative_results_plot(X, y) -> None:
    '''  
        Function: comparative_results_plot
        Parameters: 1 pd.Dataframe, 1 pd.Series
            X: features
            y: target
        Returns: None

        This function will create a plot that compares R^2 for various test sizes using the two
            different trian/test split fucntions
    '''
    # generate an array of diffenent test sizes
    test_sizes = np.arange(1,7.5)/10

    # execute regression with sklearn train_test_split
    results_sklearn = slr_var_test_size(X, y, test_sizes, True)

    # execute regression with house made train_test_split
    results_simple = slr_var_test_size(X, y, test_sizes, False)

    # plot results
    plt.figure(figsize=(10,6))
    sns.set_style('whitegrid')
    sns.lineplot(x = test_sizes, 
                 y = results_sklearn, 
                 label = 'Sklearn Train/Test Split')
    sns.lineplot(x = test_sizes, 
                 y = results_simple, 
                 label = 'House Made Train/Test Split')
    plt.title(r'$R^2$ With Respect to Test Sizes & Train/Test Split Methods', 
              weight = 'bold')
    plt.xlabel('Test Size (Proportion)')
    plt.ylabel(r'Coefficient of Determination ($R^2$)')

    # save figure
    save_path = ('figs/score_difference_plot.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
 
    plt.close();

######################################### Sample Differnces Graphic ############################################
def sample_difference_plot(X: pd.DataFrame, y: pd.Series) -> None:
    '''
        Function: sample_difference_plot
        Parameters: 1 pd.DataFrame, 1 pd.Series,
            X: feature matrix
            y: target
        Returns: None

        This function will create a visualization of the how the house made test train split
        and sklearn test trian split differ.
    '''
    # test train split sklearn
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(X, y, test_size= .3)

    # house made train test split
    X_train, X_test, y_train, y_test = simple_train_test_split(X, y)

    # generate plot
    fig, axes = plt.subplots(1,2, figsize = (10, 5))
    fig.suptitle('Sampling Differences', weight = 'bold', fontsize = 20, style = 'italic')
    sns.scatterplot(x = X_train_sk.flatten(), 
                    y = y_train_sk, 
                    color = 'firebrick', 
                    marker = '.', 
                    alpha = .5, 
                    label = 'Training Data', 
                    ax = axes[0])
    sns.scatterplot(x = X_test_sk.flatten(), 
                    y = y_test_sk, 
                    color = 'steelblue', 
                    marker = '.', 
                    alpha = .5, 
                    label = 'Test Data', 
                    ax = axes[0])
    axes[0].set_title('Sklearn Test Train Split', 
                    weight = 'bold')
    axes[0].set_xlabel('Log(Carat)', 
                       weight = 'bold')
    axes[0].set_ylabel('Log(Price)', 
                       weight = 'bold')
    sns.scatterplot(x = X_train.flatten(), 
                    y = y_train, 
                    color = 'firebrick', 
                    marker = '.', 
                    alpha = .5, 
                    label = 'Training Data', 
                    ax = axes[1])
    sns.scatterplot(x = X_test.flatten(), 
                    y = y_test, 
                    color = 'steelblue', 
                    marker = '.', 
                    alpha = .5, 
                    label = 'Test Data', 
                    ax = axes[1])
    axes[1].set_title('House Made Test Train Split', 
                    weight = 'bold')
    axes[1].set_title('House Made Test Train Split', 
                    weight = 'bold')
    axes[1].set_xlabel('Log(Carat)', 
                       weight = 'bold')
    axes[1].set_ylabel('Log(Price)', 
                       weight = 'bold')
    plt.tight_layout()

    # save figure
    save_path = ('figs/sampling_differences_plot.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
 
    plt.close();

################################################################################################
########################################### Main ###############################################
################################################################################################

def main():

    # get data: this function is defined in my_utils
    df = get_diamonds_csv()

    # log_scale both variables
    X, y = get_scale_vars(df, 'carat', 'price')

    # extract and explanatory and response variable
    X = X.to_numpy().reshape(-1, 1)
    y = y.to_numpy()

    # compare scores with different using different sampleing functions
    comparative_results_plot(X, y)

    # create sample difference plot
    sample_difference_plot(X, y)
    

if __name__ == "__main__":
    main()