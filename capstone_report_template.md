# Machine Learning Engineer Nanodegree
## Capstone Project
César Trevisan  
April 24st, 2018

## I. Definition

### Project Overview

Cancer occurs as a result of mutations, or abnormal changes, in the genes responsible for regulating the growth of cells and keeping them healthy. The genes are in each cell’s nucleus, which acts as the “control room” of each cell. Normally, the cells in our bodies replace themselves through an orderly process of cell growth: healthy new cells take over as old ones die out. But over time, mutations can “turn on” certain genes and “turn off” others in a cell. That changed cell gains the ability to keep dividing without control or order, producing more cells just like it and forming a tumor.

The term “breast cancer” refers to a malignant tumor that has developed from cells in the breast. Usually breast cancer either begins in the cells of the lobules, which are the milk-producing glands, or the ducts, the passages that drain milk from the lobules to the nipple. Less commonly, breast cancer can begin in the stromal tissues, which include the fatty and fibrous connective tissues of the breast.

Identifying correctly whether a tumor is benign or malignant is vital in deciding what is the best treatment, saving and improving the quality of life.  In this project, we used [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) to create a model able to predict if a tumor is or not dangerous based in characteristics that were computed from a digitized image of a [fine needle aspirate (FNA)](https://www.insideradiology.com.au/breast-fna/) of a breast mass.

Source: [BreastCancer.org](http://www.breastcancer.org/), [Inside Radiology](https://www.insideradiology.com.au/breast-fna/)


### Problem Statement
A tumor can be benign (not dangerous to health) or malignant (has the potential to be dangerous). Benign tumors are not considered cancerous: their cells are close to normal in appearance, they grow slowly, and they do not invade nearby tissues or spread to other parts of the body. Malignant tumors are cancerous. Left unchecked, malignant cells eventually can spread beyond the original tumor to other parts of the body. As the physical aspects of the malignant tumor differ from the benign tumor cells, we can measure the physical characteristics such as radius (mean of distances from center to points on the perimeter), texture (standard deviation of gray-scale values), perimeter, area, smoothness, compactness, concavity, concave points, symmetry or fractal dimension to understand and create two classes of tumor and identify which class each tumor belongs for new samples. 

### Metrics

In tumor cells classification is important to avoid false negatives because if a malignant tumor is predict as benign the patient will not receive treatment. That's why F1 is a ideal metric to score our model.

recall = True positive / (True positive + False negative)

precision = True positive / (True positive + False positive)


<img src="img\f12.jpg" width="300">

The final model I will analyze the model performance ploting a confusion matrix and score model using [F1 Score](<https://en.wikipedia.org/wiki/F1_score>) that is a methot to measure performance of binary classification, the F1 score is a measure of a test's accuracy, is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.


![F1 Score](conf_mat.png)


Using [F1 Score](https://en.wikipedia.org/wiki/F1_score) formula.

In statistical analysis of binary classification, the F1 score  is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall), where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.


## II. Analysis

### Data Exploration

To create a model capable to predict whether a tumor cell is or not malignant I used a labeled dataset:

[The Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+Diagnostic)  was created in 1995 by:   

Dr. William H. Wolberg (General Surgery Dept., University of  Wisconsin  Clinical Sciences Center),  
W. Nick Street (Computer Sciences Dept., University of Wisconsin)   
and  Olvi L. Mangasarian (Computer Sciences Dept., University of  Wisconsin).    
  
  Features are computed from a digitized image of a [fine needle aspirate (FNA)](https://www.insideradiology.com.au/breast-fna/) of a breast mass.  They describe characteristics of the cell nuclei present in the image.

The dataset have the following structure:
- Number of instances: 569 

 - Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

 - Diagnosis (M = malignant, B = benign)
 
 - Missing attribute values: none
 
 - Class distribution: 357 benign, 212 malignant
 
 - All feature values are recoded with four significant digits.
 
 - Ten real-valued features are computed for each cell nucleus:  
    a) radius (mean of distances from center to points on the perimeter)  
    b) texture (standard deviation of gray-scale values)  
    c) perimeter  
    d) area  
    e) smoothness (local variation in radius lengths)  
    f) compactness (perimeter^2 / area - 1.0)  
    g) concavity (severity of concave portions of the contour)  
    h) concave points (number of concave portions of the contour)  
    i) symmetry   
    j) fractal dimension ("coastline approximation" - 1)  
  
  The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.  For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

I used this physical characteristics as features to train a machine learning algorithms to create and evaluate models that classifies if a cell is benign or malignant based in this characteristics.


### Exploratory Visualization

Correlation between features


<img src="img\output_14_1.png" width="700">

Radius, Perimeter and Area have strong positive correlation

<img src="img\output_18_1.png" width="300"><img src="img\output_18_2.png" width="300">
<img src="img\output_18_3.png" width="300">

Radius have a  positive correlation with Concave Points

<img src="img\output_20_1.png" width="300">

Compactness, Concavity and Concave Points have strong positive correlation

<img src="img\output_22_1.png" width="300"><img src="img\output_22_2.png" width="300">
<img src="img\output_22_3.png" width="300">

Fractal Dimention have some negative correlation with Radius, Perimeter and Area  

<img src="img\output_25_1.png" width="300"><img src="img\output_25_2.png" width="300">
<img src="img\output_25_3.png" width="300">

#### Distribution of Classes

<img src="img\output_27_1.png" width="600">

Number of Benign :  357  
Number of Malignant :  212

#### Data Distribution 

<img src="img\output_38_1.png" width="700">


### Algorithms and Techniques

I tested many data manipulation techniques in dataset, I measured result to understand which combination of approaches is more effective 

Feature engineering: I decided create two new features,

Mean Volume

```python
# Creating a empty list
mean_volume = []
# defining pi
pi = 3.1415

# calculatin mean volume for each mean radius and saving result in mean_volume list
for i in range(len(X)):
    #aving result in mean_volume list
    mean_volume.append((math.pow(X["radius_mean"][i], 3)*4*pi)/3)

# Creating a new feature
X["mean_volume"]= mean_volume   
```

Measurements Sum

```Python
# Creating a new feature adding up some phisical measuraments
X["mesuraments_sum_mean"] = X["radius_mean"] + X["perimeter_mean"] + X["area_mean"]
```

Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, the majority of classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.[1]

[Feature Scaling - Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)



#### Feature distribution after feature scaling



<img src="img\output_37_1.png" width="700">


<img src="img\output_39_1.png" width="700">

Detect Outliers using [Tukey Method](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/)


```
# For each feature find the data points with extreme high or low values
    for feature in X.keys():

        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(X[feature], 25)

        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(X[feature], 75)

        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = (Q3 - Q1) * distance

        outliers.append(X[~((X[feature] >= Q1 - step) & (X[feature] <= Q3 + step))].index.values)
```

[Principal component analysis (PCA)](<https://en.wikipedia.org/wiki/Principal_component_analysis>) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.

#### 2-dimensional PCA Feature importance

<img src="img\output_47_1.png" width="700">

#### 2-dimensional PCA data separation

<img src="img\output_48_1.png" width="700">

#### 3-dimensional PCA data distribution

<img src="img\output_51_0.png" width="700">


Dataset has more samples of benignant tumors over malignants, it cam make classification algorithm tend to predict more cases to dominant class, I studied impact of generate new samples with three different methods.


#### [Imbalanced Learning](http://contrib.scikit-learn.org/imbalanced-learn/stable/)

***Naive random over-sampling***  
One way to fight this issue is to generate new samples in the classes which are under-represented. The most naive strategy is to generate new samples by randomly sampling with replacement the current available samples. The RandomOverSampler offers such scheme:

![](img\output_55_2.png)

**From random over-sampling to SMOTE and ADASYN**  

Apart from the random sampling with replacement, there is two popular methods to over-sample minority classes: (i) Synthetic Minority Oversampling Technique (SMOTE) and (ii) Adaptive Synthetic (ADASYN) sampling method. These algorithm can be used in the same manner:

SMOTE

![](img\output_57_2.png)

ADASYN

![](img\output_58_2.png)

### Feature Selection using [Scikit Learn](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest)

Feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator. SelectKBest removes all but the k highest scoring features





We going to performe [Feature Selection](https://en.wikipedia.org/wiki/Feature_selection) using [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest)  module in sklearn combination of features. 

Best Features using k=5:

![](img\output_40.png)


We going to performe [Feature Selection](https://en.wikipedia.org/wiki/Feature_selection) using tools like [feature_selection](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection), [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) or [SelectPercentile](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile), modules in sklearn. And [Dimension Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) to get the optiomal decision bondary/surface mesuring accuracy of different models, combination of features and dimensionality. 

### Supervised Learning

Using a dataset with 32 features going to predict if class a sample belongs, it means that we face a multidimensional classification problem. Our job is understand and prepare data to feed a algorithm wich should find the best [decision boundary](https://en.wikipedia.org/wiki/Decision_boundary) or in our case, the best decision [hypperplane](https://en.wikipedia.org/wiki/Hyperplane), see:

** Decision Boundary 2-dimensional **  

<img src="img\decision_boundary.png" width="700">

When we have 2-dimensional data we can separate data points using a decision boundary.  
  
  
** Decision Surface - 3 dimensional **  

<img src="img\decision_surface.png" width="700">

Decision surface, a surface that separates data points in a 3-dimensional space.  

The task of selected algorithm is learn from train data to find the best decision surface and correctly predict "M" Malignant or "B" Benignant for unsee test data, that was picked randomly from dataset.

### Machine Learning Algorithms

In this project I used 9 different classifiers to analyze they results and get the higher F1 score, I will give a briefly explanation about each one of this classifiers:

#### Decisions Trees

<img src="img\output_110.png" width="700">

A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

[Decision trees](<https://en.wikipedia.org/wiki/Decision_tree>) are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.


[Random Forest](<https://en.wikipedia.org/wiki/Random_forest>) or random decision forests are an [ensemble learning method](<http://scikit-learn.org/stable/modules/ensemble.html>) for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.

[Extra DecisionTrees](<http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>) This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.

[Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

[AdaBoost Classifier](https://en.wikipedia.org/wiki/AdaBoost) is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire, who won the 2003 Gödel Prize for their work. It can be used in conjunction with many other types of learning algorithms to improve performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier. AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers. AdaBoost is sensitive to noisy data and outliers. In some problems it can be less susceptible to the overfitting problem than other learning algorithms. The individual learners can be weak, but as long as the performance of each one is slightly better than random guessing, the final model can be proven to converge to a strong learner.


[XGBoost Classifier](<https://en.wikipedia.org/wiki/Xgboost>) initially started as a research project by Tianqi Chen as part of the Distributed (Deep) Machine Learning Community (DMLC) group. Initially, it began as a terminal application which could be configured using a libsvm configuration file. After winning the Higgs Machine Learning Challenge, it became well known in the ML competition circles. Soon after, the Python and R packages were built and now it has packages for many other languages like Julia, Scala, Java, etc. This brought the library to more developers and became popular among the Kaggle community where it has been used for a large number of competitions

#### Support Vector Machines

<img src="img\svm.png" width="700">

[Support Vector](<https://en.wikipedia.org/wiki/Support_vector_machine>) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
 
In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the [kernel trick](<https://en.wikipedia.org/wiki/Kernel_method>), implicitly mapping their inputs into high-dimensional feature spaces.
 

[SGD Classifier](<http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>) this estimator implements regularized linear models with [stochastic gradient descent](<https://en.wikipedia.org/wiki/Stochastic_gradient_descent>) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning, see the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance.

This implementation works with data represented as dense or sparse arrays of floating point values for the features. The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM).



#### [Logistic Regression](<https://en.wikipedia.org/wiki/Logistic_regression>)

<img src="img\logistic.png" width="700">

Was developed by statistician David Cox in 1958. The binary logistic model is used to estimate the probability of a binary response based on one or more predictor (or independent) variables (features). It allows one to say that the presence of a risk factor increases the odds of a given outcome by a specific factor. The model itself simply models probability of output in terms of input, and does not perform statistical classification, though it can be used to make a classifier, for instance by choosing a cutoff value and classifying inputs with probability greater than the cutoff as one class, below the cutoff as the other.


## Benchmark

To realize how much effective the method is, after optimize our model I'll compare results with results extracted of the following Paper: 

[Approximate Distance Classification](https://www.cs.tufts.edu/~cowen/adclass.pdf)  
Adam H. Cannon,  Lenore J. Cowen,  Carey E. Priebe  
Department of Mathematical Sciences  
The Johns Hopkins University  
Baltimore, MD, 21218  

They used k-nearest neighbor to predict cancer in same dataset to diagnosis Breast Cancer and their effectiveness, in this way we can measure and compare results in the same domain.

" The best among these was 5 nearest neighbors, and it achieved only 93.1% classification
rates. Our best reported result is 95.6% compared to a reported best known result of 97.5%" - **Approximate Distance Classification, pg.4**


## III. Methodology

### Data Preprocessing

Data structure

![](img\output_41.png)

```
# Save labels in y
y = data["diagnosis"]
```

"diagnosis" is our target, I saved this feature in a 1-dimensional dataset named y

```
# Drop columns
X = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
```

"Unannamed: 32" feature has only NaN (not a number) values
"Id" feature hasn't information to help us classify benignant/malignant tumors. I deleted "Unannamed: 32" and "Id]' features and save in a X dataframe.


Using [Train Test Split](<http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>) I divided data in train and test data.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```


### Implementation

Predict if a tumor is malignant or benignant is a classification problem, I choose some of the best classifier algorithms to perform this task. The initial process was very simple, using previously shuffled and divided data:

**X_train** (train features) and **y_train** (train labels)

I used [Scikit Learn](<http://scikit-learn.org/stable/>) that is nativily instaled from [Anaconda](<https://anaconda.org/anaconda/python>).

```
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
```

And also [installed](<http://xgboost.readthedocs.io/en/latest/build.html>) the [XGBoost Classifier](<http://xgboost.readthedocs.io/en/latest/>) 


```
from xgboost import XGBClassifier

```
Classifiers list:

```
# Random Forest Classifier 
RF_clf = RandomForestClassifier()

# Extra Trees Classifier
XT_clf = ExtraTreesClassifier()

# Decision Tree Classifier
DT_clf =DecisionTreeClassifier()

# Support Vector Machine Classifier
SV_clf = svm.SVC()

# AdaBoost Classifier
AD_clf = AdaBoostClassifier()

# Gradient Boosting Classifier
GB_clf = GradientBoostingClassifier()

# SGD Classifier
SG_clf = SGDClassifier()

# Logistic Regression
LR_clf = LogisticRegression()

# XGB Classifier
XB_clf = XGBClassifier()

classifiers = [RF_clf, XT_clf, DT_clf, SV_clf, AD_clf, GB_clf, SG_clf, LR_clf, XB_clf]

classifiers_names = ['Random Forest      ', 'Extra DecisionTrees', 'Decision Tree      ',
                     'Support Vector     ', 'AdaBoost Classifier', 'Gradient Boosting  ',
                     'SGD Classifier     ', 'Logistic Regression', 'XGB Classifier     ']

parameters = [RF_par, XT_par, DT_par, SV_par, AD_par, GB_par, SG_par, LR_par, XB_par]

```


I feed classifiers with (X, y) using the following function to train, tune using [Grid Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), print F1 scores and store results:

```
def tune_compare_clf(X, y, classifiers, parameters, classifiers_names):
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    

    print "\n" "Train size : ", X_train.shape, " and Train labels : ", y_train.shape, "\n"

    print "Test size: ", X_test.shape, " and Test labels : ", y_test.shape, "\n", "\n"
    
    results = []
    
    print "  ---- F1 Score  ----  ", "\n"

    for clf, par, name in itertools.izip(classifiers, parameters, classifiers_names):
        # Store results in results list
        clf_tuned = GridSearchCV(clf, par).fit(X_train, y_train)
        y_pred = clf_tuned.predict(X_test)
        results.append(y_pred)   

        print name, ": %.2f%%" % (f1_score(y_test, y_pred, average='weighted') * 100.0)

    result = pd.DataFrame.from_records(results)   
    
    return result, X_test,  y_test
    
```

Without any manipulation technique as scaling or outliers removing I get the following results:

Random Forest       : 95.60%  
Extra DecisionTrees : 96.47%  
Decision Tree       : 94.74%  
Support Vector      : 47.80%  
AdaBoost Classifier : 96.49%  
Gradient Boosting   : 94.74%  
SGD Classifier      : 87.42%  
Logistic Regression : 96.47%  
XGB Classifier      : 96.47%  

### Refinement

Using ** [Grid Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)** I did a Exhaustive search over specified parameter values for an estimator. The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.


```python
# Random Forest Classifier Parameters to tune
RF_par = {"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [2, 3, 10], 
          "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}

# Extra Trees Classifier Parameters to tune
XT_par = { 'n_estimators': [5, 10, 16], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10]}

# Decision Tree Classifier Parameters to tune
DT_par = { 'splitter': ['best', ], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10]}

# Support Vector Machine Classifier Parameters to tune
SV_par = {'kernel': ['rbf'], 'C': [1]}

# AdaBoost Classifier Parameters to tune
AD_par = {'n_estimators':[10, 20, 50, 60], 'learning_rate':[0.1, 0.5, 1.0, 1.5], 'algorithm':['SAMME.R', 'SAMME']}

# Gradient Boosting Classifier Parameters to tune
GB_par = {'loss':['deviance', 'exponential'], 'learning_rate':[0.01, 0.1, 0.5, 1.0], 'n_estimators':[50, 100, 150], 
          "min_samples_split": [2, 3], "min_samples_leaf": [1, 3], 'max_depth':[2, 3, 5]}

# SGD Classifier Parameters to tune
SG_par = {'loss':['hinge', 'log', 'squared_hinge', 'perceptron'], 'penalty':['l2', 'l1'], 
          'alpha':[0.00001, 0.0001, 0.001], 'epsilon':[0.01, 0.1, 0.5]}

# Logistic Regression Parameters to tune
LR_par= {'penalty':['l1','l2'], 'C': [0.5, 1, 5, 10], 'max_iter':[50, 100, 150, 200]}

# XGB Classifier Parameters to tune
XB_par = {'max_depth':[2, 3, 5], 'learning_rate':[0.01, 0.1, 0.5, 1], 'n_estimators':[50, 100, 150, 200], 'gamma':[0, 0.001, 0.01, 0.1]}
```

I defined functions to apply data manipulation techniques cited before and tested different approaches measuring results to find the best classifier configuration.

**scaler(X)** : The Function receive a Dataframe and return a Scaled Dataframe

**selector(X, y, k)** : The function receive features and labels (X, y) and a target number to select features (k) and return a new dataset wiht k best features

**remove_outliers(X, y, f, distance)**: The Function receive Features (X) and Label (y) a frequency (f) and Inter-Quartile distance (distance), and return features and labels without outliers (good_X, good_y)

**resample(X, y, method)**: The function receive features and labels (X, y) and a method to balance data available methods RandomOverSampler, ADASYN, SMOTE. The funcion returns X_resampled, y_resampled

Now we'll test using: 

**tune_compare_clf(X, y, classifiers, parameters, classifiers_names)**: a function that tune each algorith to given data and print F1 Scores. 


## IV. Results


### Model Evaluation and Validation

To evaluate results I used unsee data (20% of samples randomly chosen from original dataset) to compare results with true result (true label) and calculate precision, recall and F1 score.

#### Results using each data manipulation technique

<img src="img\output_42.png" width="700">

We can notice that scale (0 to 1) data give us more consistent results across all classifiers, specially with support vector machine that performs poorly in all scenarios without scaled data.

Remove outliers do not show any improvements as well as feature selection that improved some classifier but worsen other results.

Re-sample techniques to correct the unbalanced dataset shows improvements for most of the classifiers and random re-sample seems to be the better method than Smote and Adasyn.


#### Results using combination of data manipulation technique

<img src="img\output_43.png" width="700">

A combination of Scaling, outliers removal and re-sample was the best approach, giving us the highest scores across all classifiers and more reliable results.

The **tune_compare()** function store prediction of all algorithm in a dataframe, I used this results to create a voting system, using describe() function I get the most common prediction for all classifier creating a high resilient system and avoid fluctuations in performance.

For each sample I save in y_pred_votes the most predict class.

```
y_pred_votes = result.describe().iloc[[2]]

```

![F1 Score](conf_mat.png)

- 71 data points was correctly predicted as Benign tumors  
- 65 data points was correctly predicted as Malignant tumors  
- 01 data point was wrongly predicted as Malignan tumor  
- 01 data point was wrongly predictes as Benign tumor  

[F1 Score](https://en.wikipedia.org/wiki/F1_score):  0.9855

In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall), where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.


### Justification

In [Approximate Distance Classification](https://www.cs.tufts.edu/~cowen/adclass.pdf), Drs.
Adam H. Cannon,  Lenore J. Cowen and  Carey E. Priebe  reported the following results:

" The best among these was 5 nearest neighbors, and it achieved only 93.1% classification
rates. Our best reported result is 95.6% compared to a reported best known result of 97.5%" - **Approximate Distance Classification, pg.4**

Our best result was 99.30% using F1 scoring method to score a Extra Decision Tree Algorithm and balanced data using Random method. And using "voting system" cited before I got a F1 score of 98.55%.

Our results were significant more accurate that our benchmark model and it prove the power of modern machine learning algorithms against similar methods that were used in [Approximate Distance Classification](https://www.cs.tufts.edu/~cowen/adclass.pdf) in 1998, october. 


## V. Conclusion

### Free-Form Visualization

In our final result using a "voting system" the model predicted wrongly "only" two cases as we can see in confusion matrix bellow:

![](img\output_105_2.png)

In disease classification is very important have low false negative occurrences because if a patience have a disease and a model predict that they doesn't, the patient will not receive treatment and this may worsen the clinical picture.

The highest and more consistent results across all classifiers was achieved using Scale, Removing Outliers and Balancing data. 

- 71 data points was correctly predicted as Benign tumors  
- 65 data points was correctly predicted as Malignant tumors  
- 01 data point was wrongly predicted as Malignant tumor  
- 01 data point was wrongly predicted as Benign tumor  



### Reflection

In this project I used many data manipulation techniques as feature scaling, remove outliers and balance data creating new samples. Sometimes machine learning is a black box like, is difficult understand the optimal algorithm, parameters, features and process data methods to get the best results. To work around this issue I created a function that test many parameters for algorithm I chose, and using that function I tested different data manipulation techniques, that approach allow me running over several scenarios and found some very efficients configurations.

I expected get scores similar as my benchmark model, but the final model surprisingly surpass the bests results of benchmark model. It show how powerful modern machine learning algorithms are, and how they can be used to turn better our lives improving diagnosis results with reliable predictions.

The Wisconsin Breast Cancer dataset has 30 features and is difficult realize what is and what isn't important to correctly classify a tumor cell. Even more difficult is chosen a algorithm and parameters to get optimal results. I decided use tune methods to support me in hard task of test many parameters combination and create functions to help cover many data manipulation techniques analyzing different scenarios that give us exceptional results.


### Improvement

The approach I adopted in this project were try several different algorithm, parameters, methods and combinations to find the best configuration. 

I believe that combining new data manipulation techniques, more tuning parameters and algorithms best results can be achieved. But this can increase exponentially the number of train sets which may cause time consumption issues.



