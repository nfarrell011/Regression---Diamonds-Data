'''
    Problem Set 5: q1
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file will visualize the relationship between "price" and "carat" to help determine appropriate for 
    modeling of their relationship
'''
from my_utils import get_diamonds_csv 

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


def price_carat_vis(df: pd.DataFrame) -> None:
    '''
        Function: price_carat_vis
        Parameters: 1 pd.DataFrame
        Returns: None

        This function will generate an sns.JointGrid displaying the relationship between 
            price and carat in the diamonds df
    '''
    sns.set_theme(style = 'ticks')
    plot = sns.JointGrid(data = df, 
                         x = 'carat', 
                         y = 'price', 
                         marginal_ticks = True)
    plot.plot_joint(sns.scatterplot, marker = '.')
    plot.plot_marginals(sns.histplot,
                        kde = True)
    plot.set_axis_labels(ylabel = 'Price ($)', 
                         xlabel = 'Carat', 
                         weight = 'bold')
    plot.fig.suptitle('Price with Respect to Carat: \n with Marginal Histograms', 
                      weight = 'bold', 
                      fontsize = 16,
                      y = 1.01)

    # save figure
    save_path = ('figs/price_carat_plot.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
 
    plt.close();

def main():
    
    # get data
    df = get_diamonds_csv()

    # generate fig
    price_carat_vis(df)

if __name__ == "__main__":
    main()
