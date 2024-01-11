# Weights of Evidence Transformation + Auto Grouping for Classification 

The purpose of this repository is to allow for a weights of evidence transformation of variables in a Pandas dataframe when performing a single class classification task. This type of transformation is often used in credit scoring models however can be used in a variety of applications. A weights of evidence transformation is typically performed on discrete or categorical variables, however this repository is able to handle numeric variables by providing an auto-grouping functionality.

## How to Use:
When working with only categorical variables, the following can be done:
```python
import WeightsOfEvidence
import pandas as pd

df = pd.read_csv('...')
target = '...'
columns = [col for col in df.columns if col != target]

woe = WeightsOfEvidence(df, columns, target)
```
The transformed columns will then be available in ```woe.df``` and have a ```_woe``` suffix.

When numeric variables are available, grouping is performed. An example can be seen below:
```python
woe = WeightsOfEvidence(df, columns, target, grouping_method='equal_bins')
```
This will then create transformed columns as previously mentioned but also produce numeric variables with a ```_grouped``` suffix in ```woe.df```. 
Optimal splitting through decision tree classification can be performed using ```grouping_method='decision_tree'```

## Plotting Weights of Evidence Distributions
The weights of evidence for all columns can be plotted using ```woe.plot_woe()``` to produce something like the following:
![image](https://github.com/VassMorozov/Weights-of-Evidence/assets/28609388/8b59bbc9-437a-4e12-9f4f-d68e631c1b79)

This example shows the plot for a categorical and numeric variable, where the numeric variable was grouped as previously described.

## Notes
* No missing values should be present in either the columns to transform or the target variable. This class expects missings to be treated prior to calling.
* Future releases may incorporate additional grouping mechanisms.
