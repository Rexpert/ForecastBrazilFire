import pandas as pd
import matplotlib.pyplot as plt


def plot_graph(df):
    df.iloc[:, 2].plot()
    plt.title('Student CPI Data 2000-2016')
    plt.show()


def show_na(df):
    # check na values
    return df.isna().sum()
