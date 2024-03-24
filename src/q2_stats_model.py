'''
    Problem Set 5: q2
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file will execute an SLR using statsmodel.api with price and carat both log scaled
'''
from my_utils import get_diamonds_csv
from my_utils import get_scale_vars

import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd  

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm  
from scipy.stats import norm 

##################################### Stats Models: SLR ######################################
def stats_model_SLR(features: pd.DataFrame, response: pd.DataFrame):
    '''
        Function: stats_model_SLR

        Parameters: 2 pd.DataFrames
            features: the design matrix
            reponse: the reponse variable

        Returns: 1 float, 1 statsmodels.regression.linear_model.RegressionResultsWrapper
            r_squared: the coefficient of determination of the linear model
            results: results object of linear model

        This function will fit a stats models linear regression
    '''
    # add y-intercept column
    features['constant'] = 1

    # fit linear model, and save results
    results = sm.OLS(response, features).fit()

    # extract r^2
    r_squared = results.rsquared

    # return r^2 and results
    return r_squared, results

##################################### Generate DF of Results ######################################

def create_fitted_residuals_df(results, X, y) -> None:
    '''  
        Function: create_fitted_residuals_df

        Parameters: 1 statsmodels.regression.Wrapper, 1 pd.Dataframe, 1 pd.Series
            results: the results object from the linear model
            X: feature matrix
            y: response variable

        Returns: 1 pd.Dataframe
            fitted_residuals_df: a data with the predictor, response, fitted values, and residuals

        This function will pull the betas from the model results, compute y_hat and the residuals, and put the
        results in a dataframe
    '''
    # extract params
    beta_0 = results.params.iloc[1] # intercept
    beta_1 = results.params.iloc[0] # coefficient

    # compute fitted values
    y_hat = beta_0 + (beta_1 * X.carat)

    # compute residuals
    residuals = y - y_hat

    # generate df with x, y, y_hat, and residuals
    fitted_residuals_df = pd.DataFrame({'x': X.carat,
                                        'y': y,
                                        'y_hat': y_hat,
                                        'residuals': residuals})
    # return df
    return fitted_residuals_df

##################################### Visualize Results ######################################

def visualize_results(fitted_residuals_df: pd.DataFrame) -> None:
    '''  
        Function: visualize_results

        Parameters: 1 pd.Dataframe
            fitted_residuals_df: a data with the results from linear model
            
        Returns: None

        This function will create a visualization of the observed versus fitted values
    '''
    # extract sample from fitted_residuals_df
    sampled_df = fitted_residuals_df.sample(n = 500, random_state = 42)

    # plot observed versus fitted values
    plt.figure(figsize=(10, 8))
    plt.vlines(sampled_df.x, 
               ymin = sampled_df.y, 
               ymax = sampled_df.y_hat, 
               color = 'k', 
               linewidth = .5)
    plt.plot(sampled_df.x, 
             sampled_df.y_hat, 
             'b', 
             alpha = .5, 
             label = r"$\hat{y}$")
    plt.scatter(sampled_df.x, 
                sampled_df.y, 
                c = "r", 
                s = 10, 
                label = "y")
    plt.ylabel(r'$\log(Carat)$', 
               weight = 'bold')
    plt.xlabel(r'$\log(price)$', 
               weight = 'bold')
    plt.title(f'Observed Versus Fitted Values \n Sample Size = 500', 
              weight = 'bold', 
              fontsize = 20)
    plt.legend()

    # save figure
    save_path = ('figs/fitted_observed_plot.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
 
    plt.close();

##################################### Visualize Residuals ######################################

def visualize_residuals(fitted_residuals_df: pd.DataFrame) -> None:
    '''  
        Function: visualize_residuals

        Parameters: 1 pd.Dataframe
            fitted_residuals_df: a data with the results from linear model
            
        Returns: None

        This function will create a visualization of the behavior of the residuals
    '''
    # extract sample from fitted_residuals_df
    sampled_df = fitted_residuals_df.sample(n = 500, random_state = 42)

    # normalize the residuals
    residuals = fitted_residuals_df.residuals.to_numpy().reshape(-1, 1)
    res_scaled = StandardScaler().fit_transform(residuals)

    # generate standard normal distribution using scipy
    scipy_norm = norm.rvs(0, 1, size = 53940)

    # plot behavior of the residuals
    fig, axes = plt.subplots(3, 1, figsize = (7, 15))
    fig.suptitle('Conditions Check: Residuals', weight = 'bold', fontsize = 20)
    axes[0].axhline(0, c='k');
    sns.scatterplot(data = fitted_residuals_df, 
                    x = 'y_hat', 
                    y = 'residuals', 
                    ax = axes[0], 
                    marker = '.', 
                    alpha = .5)
    axes[0].set_title('Residuals v. Fitted Values: \n Full Data Set', 
                    weight = 'bold', 
                    style = 'italic')
    axes[0].set_xlabel(r'$\hat{y}$', weight = 'bold')
    axes[0].set_ylabel('Residuals', weight = 'bold')
    sns.scatterplot(data = sampled_df, 
                    x = 'y_hat', 
                    y = 'residuals', 
                    ax = axes[1], 
                    marker = '*', 
                    color = 'firebrick')
    axes[1].axhline(0, c='k')
    axes[1].set_title('Residuals v. Fitted Values: \n Sample Size = 500', 
                      weight = 'bold', 
                      style = 'italic')
    axes[1].set_xlabel(r'$\hat{y}$', 
                       weight = 'bold')
    axes[1].set_ylabel('Residuals', 
                       weight = 'bold')
    sns.histplot(res_scaled, 
                 color = 'blue', 
                 alpha = .7, 
                 ax = axes[2], 
                 label  = r'Residuals Normalized ~ $\frac{x-\mu}{\sigma}$', 
                 legend = None)
    sns.histplot(scipy_norm, 
                 color = 'red', 
                 alpha = .3, 
                 ax = axes[2], 
                 label = f'Gaussian Normal Distribution')
    axes[2].set_title('Distribution of Residuals: \n Gaussian Overlay', 
                      weight = 'bold', 
                      style = 'italic')
    axes[2].set_xlabel('Residiuals', 
                       weight = 'bold')
    axes[2].set_ylabel('Count', 
                       weight = 'bold')
    fig.legend(bbox_to_anchor=[.51, .29])
    plt.tight_layout()
    
    # save figure
    save_path = ('figs/residuals_plot.png')
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

    # execute model
    stats_model_r_squared, results = stats_model_SLR(X, y)

    # display results
    print(results.summary())
    print(f'The coefficient of determination for the stats models linear model is {stats_model_r_squared:.4f}.')

    # create dataframe with the fitted vals and residuals
    fitted_residuals_df = create_fitted_residuals_df(results, X, y)

    # visualize results
    visualize_results(fitted_residuals_df)

    # visualize residuals
    visualize_residuals(fitted_residuals_df)

if __name__ == "__main__":
    main()

