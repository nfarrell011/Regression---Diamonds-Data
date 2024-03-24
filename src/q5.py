'''
    Problem Set 5: q5
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023
'''
from my_utils import get_scale_vars
from my_utils import get_diamonds_csv

import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd 
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, r2_score



#################################### Find Best Model Function ##########################################
def find_best_model(col: pd.Series):
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
    # read in dataframe
    df = get_diamonds_csv() # function defined in my_utils

    # extract and scale features and target
    X, y = get_scale_vars(df, 'carat', 'price') # function defined in my_untils

    # select the catagorical variable to encode
    column_to_encode = df[[col]]
    
    # instaniate encoder
    encoder = OneHotEncoder(drop = 'first')

    # encode catagorical variable
    enc_data = pd.DataFrame(encoder.fit_transform(column_to_encode).toarray()) 

    # concat encoded variable to feature matrix
    X = pd.concat([X, enc_data], axis = 1)

    # change col names to strings
    X.columns = X.columns.astype(str)

    # instantiate model
    reg = LinearRegression(fit_intercept = True)

    # define the splits
    #cv = ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 42)

    r2 = make_scorer(r2_score)

    # execute models
    scores = pd.DataFrame(cross_validate(reg, X, y, scoring = r2, return_train_score = True))
    results = cross_validate(reg, X, y, scoring = r2, return_train_score = True, return_indices = True)

    # extract test and train scores
    test_scores = scores.test_score
    train_scores = scores.train_score

    return test_scores, train_scores, results

#################################### Generate Aggregated Scores DataFrame ##########################################
def aggregated_scores_df(test_scores: tuple, train_scores: tuple, cols: list) -> pd.DataFrame:
    '''  
        Function: aggregated_scores_df
        Parameters: 2 tuples, 1 list
            test_scores: model test scores
            train_scores: model_train_scores
            cols: column names
        Returns: 1 pd.DataFrame
            grouped_df: a dataframe with aggregated results of model performance

        This function will take the test scores and train scores from the 5 models performed with the
        3 different catagorical variables, put them in a dataframe and compute the mean and standard deviation.
    '''
    # put scores tuples into df and transpose
    df = pd.DataFrame(test_scores).T
    df_2 = pd.DataFrame(train_scores).T

    # name the columns 
    df.columns = cols
    df_2.columns = cols

    # put dfs in long form
    df = pd.melt(df, 
                 value_vars = ['cut', 'color', 'clarity'], 
                 value_name = 'test_scores', 
                 var_name = 'cat_variable')
    df_2 = pd.melt(df_2, 
                   value_vars = ['cut', 'color', 'clarity'], 
                   value_name = 'train_scores')

    # concat and drop duplicate column
    df = pd.concat([df, df_2], axis = 1).drop(columns = ['variable'])

    # groupby cat variable and find the mean and std
    grouped_df = df.groupby('cat_variable').agg({'test_scores': ['mean', 'std'], 
                                                 'train_scores': ['mean', 'std']}).reset_index()

    return grouped_df

#################################### Aggregated Scores Fig 1 ##########################################
def comparative_fig(grouped_df: pd.DataFrame) -> None:
    ''' 
        Function: caparative_fig
        Parameters: 1 pd.Dataframe
            grouped_df: the dataframe the with aggregated results
        
        Returns: None

        This function will generate a figure that will compare the mean test scores and mean train
        scores of the models incorporating different catagorical variables.
    '''
    # this is for positioning
    x_positions = np.arange(len(grouped_df.cat_variable))

    # create fig
    fig, ax = plt.subplots(figsize = (7,4))
    ax.errorbar(x_positions - .2, 
                grouped_df.test_scores['mean'], 
                yerr = grouped_df.test_scores['std'], 
                fmt = '.', 
                color = 'black',
                ecolor = 'r', 
                elinewidth=2, 
                capsize = 5, 
                label = 'Test Scores')
    ax.errorbar(x_positions + .2, 
                grouped_df.train_scores['mean'], 
                yerr = grouped_df.train_scores['std'], 
                fmt = '.', 
                color = 'black', 
                ecolor = 'blue', 
                elinewidth=2, 
                capsize = 5, 
                label = 'Train Scores')
    ax.set_title('Mean Test & Train Scores \n with Inclusion of Catagorical Variable \n Error bar = std',
            weight = 'bold', 
            fontsize = 16, 
            style = 'italic')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped_df['cat_variable'])
    ax.set_xlabel('Catagorical Variable Included in MLR', 
            weight = 'bold')
    ax.set_ylabel('Coefficient of Determination ($R^2$)', 
            weight = 'bold')
    plt.legend()
    
    # save figure
    save_path = ('figs/best_model_fig_2.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close();

#################################### Aggregated Scores Fig 2 ##########################################
def zoomed_in_comparative_fig(grouped_df: pd.DataFrame) -> None:
    ''' 
        Function: caparative_fig
        Parameters: 1 pd.Dataframe
            grouped_df: the dataframe the with aggregated results
        
        Returns: None

        This function will generate a figure that will compare the mean test scores and mean train
        scores of the models incorporating different catagorical variables (Zoomed in).
    '''
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Mean Test & Train Scores of MLR with Inclusion of Single Catagorical Variable \n Error Bar = std', 
                weight = 'bold', 
                style = 'italic', 
                fontsize = 18)
    fig.supxlabel('Catagorical Variable Included in MLR', 
                weight = 'bold')
    for i, j in enumerate(grouped_df.cat_variable.unique()):
        x = grouped_df.loc[grouped_df.cat_variable == j]
        axes[i].errorbar(x.cat_variable, 
                        x.test_scores['mean'], 
                        yerr = x.test_scores['std'], 
                        fmt = '.', 
                        color = 'black', 
                        ecolor = 'r', 
                        elinewidth = 2, 
                        capsize = 5, 
                        label = 'Test Scores')
        axes[i].errorbar(x.cat_variable, 
                        x.train_scores['mean'], 
                        yerr = x.train_scores['std'], 
                        fmt = '.', 
                        color = 'black', 
                        ecolor = 'blue', 
                        elinewidth=2, 
                        capsize = 5, 
                        label = 'Train Scores')
        axes[i].set_ylabel('$R^2$ (Zoomed)', 
                        weight = 'bold')
        axes[i].legend()

        # add annotations for mean and std test scores
        for idx, mean_score in enumerate(x.test_scores['mean']):
            std_score = x.test_scores['std'].iloc[idx]
            axes[i].annotate(f'mean: {mean_score:.4f}. \n std: {std_score:.5f}.', 
                            (x.cat_variable.iloc[idx], mean_score), 
                            textcoords='offset points', 
                            xytext=(-6, -10), 
                            ha='right', 
                            fontsize = 7, 
                            weight = 'bold')
            
        # add annotations for mean and std train scores
        for idx, mean_score in enumerate(x.train_scores['mean']):
            std_score = x.test_scores['std'].iloc[idx]
            axes[i].annotate(f'mean: {mean_score:.4f}.\n std: {std_score:.5f}.', 
                            (x.cat_variable.iloc[idx], mean_score), 
                            textcoords='offset points', 
                            xytext=(-6, -10), 
                            ha='right', 
                            fontsize = 7, 
                            weight = 'bold')
    plt.tight_layout()

    # save figure
    save_path = ('figs/best_model_fig_3.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close();

#################################### Data in Folds Fig ##########################################
def display_data_in_splits(df: pd.DataFrame, results) -> None:
    '''  
        Function: display_data_in_splits
        Parameters: 1 pd.DataFrame, 1 results object
        Returns: None

        This function will create a figure that will display the speciific datapoints used in the
        k-fold split.
    '''
    # extract the indices
    train_indices = [i for i in results[0]['indices']['train']]
    test_indices = [i for i in results[0]['indices']['test']]

    # log scale the variables
    df.price = np.log(df.price)
    df.carat = np.log(df.carat)

    # create fig
    fig, axes = plt.subplots(1,5, 
                             figsize = (12,5), 
                             sharex = True, 
                             sharey = True)
    fig.suptitle('Training and Testing Datapoints Within Each Fold', 
                 weight = 'bold', 
                 fontsize = 16)
    fig.supxlabel('Log(Carat)', weight = 'bold', style = 'italic')
    fig.supylabel('Log(Price)', weight = 'bold', style = 'italic')
    for i in range(0, 5):
        df_1 = df.loc[train_indices[i]]
        df_2 = df.loc[test_indices[i]]
        sns.scatterplot(data = df_1, 
                        x = df_1.carat, 
                        y = df_1.price, 
                        color = 'red', 
                        ax = axes[i], 
                        marker = '.', 
                        label = 'Train Data', 
                        alpha = .5)
        axes[i].set_title(f'Fold {i + 1}', 
                          weight = 'bold', 
                          style = 'italic')
        axes[i].set_ylabel(None)
        axes[i].set_xlabel(None)
        sns.scatterplot(data = df_2, 
                        x = 'carat', 
                        y = 'price', 
                        color = 'black', 
                        ax = axes[i], 
                        marker = '.', 
                        label = 'Test Data',
                        alpha = .5)
        axes[i].set_ylabel(None)
        axes[i].set_xlabel(None)
        if i != 0:
            axes[i].get_legend().remove()
        axes[0].legend().set_bbox_to_anchor((-.4, .9))
    #plt.show()
    
    # save fig
    save_path = ('figs/data_in_k_splits.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close();

################################################################################################
########################################### Main ###############################################
################################################################################################

def main():

    # read in data
    df = get_diamonds_csv()

    # these are the catagorical variables to test
    cols = ['cut', 'color', 'clarity']

    # this will invoke find_best_model for each catagorical variable
    test_scores, train_scores, results = zip(*[find_best_model(i) for i in cols])

    # this will generate a grouped df of the aggregated scores from the models
    grouped_df = aggregated_scores_df(test_scores, train_scores, cols)

    # this will create fig of the aggregated scores
    comparative_fig(grouped_df)

    zoomed_in_comparative_fig(grouped_df)

    # this will generate a figure of the data was split
    display_data_in_splits(df, results)


if __name__ == "__main__":
    main()