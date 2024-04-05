import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import plotly


def cal_shapley_values_mse(shapley1_file_name, shapley2_file_name):
    """

    Parameters
    ----------
    shapley1_file_name
    shapley2_file_name

    Returns
    -------

    """

    shapley1 = pd.read_csv(shapley1_file_name, delimiter=";").drop("Weight", axis=1)
    shapley2 = pd.read_csv(shapley2_file_name, delimiter=";").drop("Weight", axis=1)
    mse = ((shapley1.values - shapley2[shapley1.columns.values].values) ** 2).mean()

    return mse


def plotly_bar_plot_comparisons(all_permutations_shapley1_file_name,
                                limited_permutations_by_correlations_shapley2_file_name):
    """

    Parameters
    ----------
    all_permutations_shapley1_file_name
    limited_permutations_by_correlations_shapley2_file_name

    Returns
    -------

    """

    shapley1 = pd.read_csv(all_permutations_shapley1_file_name, delimiter=";").drop("Weight", axis=1)
    shapley2 = pd.read_csv(limited_permutations_by_correlations_shapley2_file_name, delimiter=";").drop("Weight",
                                                                                                        axis=1)
    shapley2 = shapley2[shapley1.columns.values]

    assert (shapley2.columns.values == shapley1.columns.values).all()

    fig = go.Figure(data=[go.Bar(name="Full permutations", x=shapley1.columns.values, y=shapley1.values.flatten()),
                          go.Bar(name="Limited permutations", x=shapley2.columns.values, y=shapley2.values.flatten())])
    fig.update_layout(barmode='group')
    plotly.offline.plot(fig, filename='results/plots/shapley_comparisons_dominator.html', auto_open=True)
    # merge two dataframes:


if __name__ == "__main__":
    shapley1_file_name = "results/out_synthetic_data.csv"
    shapley2_file_name = "results/out_with_limited_permutations_syn_data.csv"

    mse = cal_shapley_values_mse(shapley1_file_name, shapley2_file_name)
    sqrt_mse = math.sqrt(mse)
    print(f"MSE errors between shapley calculation with all possible permutations and limited permutations are:  {mse},"
          f"Mean square root error {sqrt_mse}")
    plotly_bar_plot_comparisons(shapley1_file_name, shapley2_file_name)
