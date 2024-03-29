import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class Grouping(object):

    """
        The purpose of this class is to split numeric variables for single or multi class classification tasks.

        Keyword arguments:
            df (pandas.core.frame.DataFrame): Dataframe containing numeric variables to split
            numeric_columns (list): List of numeric columns to split
            target (str): Name of column in df containing target variable
            method (str): Preferred grouping method, either 'decision_tree' or 'equal_bins'
            depth (int): depth of decision trees if grouping_method == 'decision_tree'
            min_samples (int): number of samples in decision tree leaves if grouping_method == 'decision_tree'
            random_state (int): random seed if grouping_method == 'decision_tree'
            n_bins (int): number of bins wanted if grouping_method == 'equal_bins'
            add_data (pandas.core.frame.DataFrame): additional data to apply groupings calculated on df to

        Attributes:
            df (pandas.core.frame.DataFrame): Dataframe containing grouped versions of columns.
            cols_to_group (list): Columns to group

        Methods:
            decision_tree_grouping(self, df, target, depth=None, min_samples=1, random_state=None):
                Splits numeric variables using binary decision trees.

            binning(self, df, n_bins):
                Splits numeric variables based on equal bin sizes.
        """

    def __init__(self, df, numeric_columns, target, method='decision_tree', depth=2, min_samples=1,
                 random_state=None, n_bins=2, add_data=None):

        self.cols_to_group = numeric_columns
        self.add_data = add_data

        if self.add_data is not None:
            assert set(self.cols_to_group).issubset(self.add_data.columns), \
                'Columns to group are not in additional data.'
            assert (self.add_data.loc[:, self.cols_to_group].isnull().sum() > 0).sum() == 0, \
                'Missing values found in columns to group for additional data. Deal with these before ' \
                'calling transformation.'
            self.add_data = add_data

        if method == 'decision_tree':
            self.df = self.decision_tree_grouping(df, target, depth=depth, min_samples=min_samples,
                                                  random_state=random_state)

        if method == 'equal_bins':
            self.df = self.binning(df, n_bins)

    def decision_tree_grouping(self, df, target, depth=None, min_samples=1, random_state=None):
        y = df.loc[:, target]

        for column in self.cols_to_group:
            # Initialize decision tree classifier
            tree_classifier = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples,
                                                     random_state=random_state)
            # Fit the decision tree model
            tree_classifier.fit(df.loc[:, column].values.reshape(-1, 1), y)

            # Assign categorical group based on which leaf the value falls into
            df[f"{column}_grouped"] = tree_classifier.apply(df.loc[:, column].values.reshape(-1, 1))

            if self.add_data is not None:
                self.add_data[f"{column}_grouped"] = tree_classifier.apply(self.add_data.loc[:, column].values.reshape(-1, 1))

        return df

    def binning(self, df, n_bins):
        for column in self.cols_to_group:
            df[f'{column}_grouped'], _bins = pd.qcut(df[column], q=n_bins, labels=[i for i in range(n_bins)],
                                                     retbins=True)

            if self.add_data is not None:
                self.add_data[f"{column}_grouped"] = pd.cut(self.add_data[column], _bins)

        return df
