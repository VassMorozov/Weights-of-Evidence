import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import copy
from grouping import Grouping


class WeightsOfEvidence(object):

    """
    The purpose of this class is to perform weights of evidence (WoE) transformations to variables in a Pandas
    dataframe for classification tasks. WoE transformations are typically used for discrete variables so where numeric
    variables are provided, this class will group them using the provided grouping method. Additional utility includes
    plotting WoE distributions.

    Keyword arguments:
        df (pandas.core.frame.DataFrame): Dataframe to have WoE transformation applied
        columns (list): List of columns in df to have WoE transformation applied
        target (str): Name of column in df containing target variable for WoE calculation
        grouping_method (str): Preferred grouping method, either 'decision_tree' or 'equal_bins'
        depth (int): depth of decision trees if grouping_method == 'decision_tree'
        min_samples (int): number of samples in decision tree leaves if grouping_method == 'decision_tree'
        random_state (int): random seed if grouping_method == 'decision_tree'
        n_bins (int): number of bins wanted if grouping_method == 'equal_bins'

    Attributes:
        df (pandas.core.frame.DataFrame): Dataframe containing transformed columns
        numeric_columns (list): List of numeric columns in df pre-grouping
        woe_dict (dict): A dictionary containing summary dataframes for each categorical and grouped numeric column

    Methods:
        calculate_woe(self, target):
            Calculates weights of evidence for all columns and stores information in summary dictionary

        assign_woe(self):
            Assigns weights of evidence to dataframe from which it was calculated

        plot_woe(self, confidence=0.975):
            Plots distribution of weights of evidence for all transformed variables as well as a summary table
    """

    def __init__(self, df, columns, target, grouping_method=None, depth=2, min_samples=1,
                 random_state=10, n_bins=2):

        # Will be an empty list if no numeric variables are being grouped
        self.numeric_columns = df.loc[:, columns].select_dtypes(include='number').columns.tolist()
        if grouping_method is not None:
            grouper = Grouping(df, self.numeric_columns, target, method=grouping_method, depth=depth,
                               min_samples=min_samples, random_state=random_state, n_bins=n_bins)
            self.df = grouper.df

            self.columns = [f'{col}_grouped' if col in self.numeric_columns else col for col in columns]
        else:
            self.columns = columns
            self.df = df

        self.woe_dict = self.calculate_woe(target)
        self.df = self.assign_woe()

    def calculate_woe(self, target):

        # Initialise dictionary of weights of evidence
        woe_dict = {}

        for col in self.columns:
            # Calculate WoE for each column
            woe_df = self.df.groupby(col, observed=False)[target].agg(['count', 'sum']).reset_index()
            woe_df.columns = [col, 'total_count', 'event_count']

            woe_df['non_event_count'] = woe_df['total_count'] - woe_df['event_count']
            # To ensure no division by 0 set to small value if count is 0
            woe_df['non_event_count'] = woe_df['non_event_count'].apply(lambda x: x if x > 0 else 1e-5)

            woe_df['event_rate'] = woe_df['event_count'] / self.df[target].sum()
            woe_df['non_event_rate'] = woe_df['non_event_count'] / (self.df.shape[0] - self.df[target].sum())
            woe_df['woe'] = (woe_df['event_rate'] / woe_df['non_event_rate']).apply(
                lambda x: 0 if x == 0 else np.log(x))
            woe_dict[col] = woe_df

        return woe_dict

    def assign_woe(self):

        for key, val in self.woe_dict.items():
            self.df[f'{key}_woe'] = self.df[key].map(val.set_index(key)['woe'])
        return self.df

    def plot_woe(self, confidence=0.975):

        z_value = st.norm.ppf(confidence)

        # Taking copy so original is not edited
        _copy = copy.deepcopy(self.woe_dict)
        for key, val in _copy.items():

            # Calculate standard error
            val['se'] = np.sqrt(1 / val['event_count'] + 1 / val['non_event_count'])
            # Calculate the margin of error
            val['moe'] = z_value * val['se']

            # Calculate the confidence interval
            val['lower_limit'] = val['woe'] - val['moe']
            val['upper_limit'] = val['woe'] + val['moe']

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=val['woe'], y=val[key], label='WoE', ax=ax)
            ax.errorbar(x=val['woe'], y=val[key], xerr=val['moe'], fmt='o', ecolor='r',
                        label='CI')

            # Create a table with count and percentage information if the variable is numeric and grouped
            if key.replace("_grouped", "") in self.numeric_columns:
                table_data = [["Category", "Lower Bound", "Upper Bound", "Count", "Exit Rate (%)"]]
            else:
                table_data = [["Category", "Count", "Exit Rate (%)"]]

            for category_value in val[key]:

                count = int(val[val[key] == category_value]['total_count'].values[0])
                dist = f"{round(val[val[key] == category_value]['event_count'].values[0] / count, 2)}%"

                # When numeric variables are grouped, plot lower and upper bound in the table
                if key.replace("_grouped", "") in self.numeric_columns:
                    lower_bound = self.df[self.df[key] == category_value].min()[key.replace("_grouped", "")]
                    upper_bound = self.df[self.df[key] == category_value].max()[key.replace("_grouped", "")]
                    table_data.append([category_value, lower_bound, upper_bound, count, dist])
                else:
                    table_data.append([category_value, count, dist])

            widths = [0.3 for i in range(len(table_data[0]))]
            table = plt.table(cellText=table_data, loc='right', cellLoc='center', colWidths=widths)
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            ax.add_table(table)

            plt.title(f'Scatterplot of WoE for {key}')
            plt.xlabel('WoE')
            plt.ylabel(key)
            plt.legend()
            plt.show()
