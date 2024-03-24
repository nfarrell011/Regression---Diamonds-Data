'''
    Problem Set 5: q2
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file will investigate logarithmic transformations to see if you can find a more appropriate relationship between 
    "price" and "carat" for univariate linear regression.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from my_utils import get_diamonds_csv

################################### Explore Log Scales Fig ########################################
def log_explore_vis(df: pd.DataFrame) -> None:
    '''
        Function: log_explore_vis
        Parameters: 1 pd.DataFrame
        Returns: None

        This function generate scatterplots exploring the relationship between carat and price
            using different log scaling combinations
    '''
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1,3, figsize = (15,6))
    fig.suptitle('Logarithmic Scaling Exploration', 
                fontsize = '18', 
                weight = 'bold')
    sns.scatterplot(data = df, 
                    x = np.log(df.carat), 
                    y = df.price, 
                    ax = axes[0], 
                    marker = '.')
    axes[0].set_title('Carat Log Scaled', 
                    weight = 'bold', 
                    style = 'italic')
    axes[0].set_ylabel('Price')
    axes[0].set_xlabel(r'$\log(Carat)$')
    sns.scatterplot(data = df, 
                    x = df.carat, 
                    y = np.log(df.price), 
                    ax = axes[1], 
                    marker = '.')
    axes[1].set_title('Price Log Scaled', 
                    weight = 'bold', 
                    style = 'italic')
    axes[1].set_ylabel(r'$\log(Price)$')
    axes[1].set_xlabel('Carat')
    sns.scatterplot(data = df, 
                    x = np.log(df.carat), 
                    y = np.log(df.price), 
                    ax = axes[2], 
                    marker = '.')
    axes[2].set_title('Carat & Price Log Scaled', 
                    weight = 'bold', 
                    style = 'italic')
    axes[2].set_ylabel(r'$\log(Price)$')
    axes[2].set_xlabel(r'$\log(Carat)$')
    plt.tight_layout();

    # save figure
    save_path = ('./figs/explore_log_plot.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
 
    plt.close();

def main():
    
    # get data: this function is defined in my_utils
    df = get_diamonds_csv()

    # generate fig
    log_explore_vis(df)

if __name__ == "__main__":
    main()