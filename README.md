# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

**Note**

The capstone project used python 2.7 and the libraries is listed bellow, that can be imported using this code bellow:


```
import pandas as pd
import numpy as np
import math
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import visuals as vs
from mpl_toolkits.mplot3d import axes3d

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree

from xgboost import XGBClassifier
```

To reproduce results I set seeds as the follow:

```

# Set Random Seed
np.random.seed(42)
np.random.RandomState(42)
```

Original data can be dowloaded in 

<https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)><https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>

