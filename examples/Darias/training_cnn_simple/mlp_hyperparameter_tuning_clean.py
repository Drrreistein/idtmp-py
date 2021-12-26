
import numpy as np

import matplotlib.pyplot as plt

from collections import defaultdict
import seaborn as sns
from sklearn import preprocessing
import pandas as pd

data = defaultdict(list)

with open("mlp_hyperparameter_tuning_clean.txt", 'r') as f:
    for l in f:
        str_list = l.split(' ')
        data[str_list[0]].append(np.float(str_list[1][:-1]))


data_pd  = pd.DataFrame(data)
data_pd['learning_rate'] = 10**data_pd['learning_rate']
data_pd['batch_size'] = 2**data_pd["batch_size"]

data_sel = data_pd[['batch_size', 'dense_layer', 'learning_rate', 'dropout', 'val_accuracy', 'val_loss']]

data_sel['batch_size'] = preprocessing.minmax_scale(data_sel['batch_size'])
data_sel['learning_rate'] = preprocessing.minmax_scale(data_sel['learning_rate'])
data_sel['dropout'] = preprocessing.minmax_scale(data_sel['dropout'])
data_sel['dense_layer'] = preprocessing.minmax_scale(data_sel['dense_layer'])

# Compute the correlation matrix
corr = data_sel.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()