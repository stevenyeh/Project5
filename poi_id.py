#!/usr/bin/python

import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA, RandomizedPCA

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
# Decided to use all features (except email_address) since I plan on using SelectKBest to select relevant features
# The 'email_address' of a possible poi only serves as an identifier and not a relevant feature.
features_list = ['poi',
                 'bonus',
                 'deferral_payments',
                 'deferred_income',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'loan_advances',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'salary',
                 'shared_receipt_with_poi',
                 'to_messages',
                 'total_payments',
                 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1 continued: Data Exploration

### Total Number of Data Points
# 146 data points
number_data_points = len(data_dict)

### Allocation across classes (POI/non-POI)
# 18 poi's, 128 non-poi's
count_poi = sum([1 if data_dict[person]['poi'] == True else 0 for person in data_dict])
count_non_poi = sum([1 if data_dict[person]['poi'] != True else 0 for person in data_dict])

### Number of features
# 21 features total
keys = []
for key, value in data_dict['METTS MARK'].iteritems():
    keys.append(key)
number_features = len(keys)

### Features with many missing values
# Searching for features with over half of total individuals with 'NaN' values
# ['deferral_payments', 'restricted_stock_deferred', 'loan_advances', 'director_fees', \
# 'deferred_income', 'long_term_incentive']

features_missing_values = []
for name in data_dict:
    for feature in data_dict[name]:
        if feature not in features_missing_values:
            count = 0
            for person in data_dict:
                if data_dict[person][feature] == 'NaN':
                    count += 1
            if count >= number_data_points / 2:
                features_missing_values.append(feature)

### Attempting to find features that contain a lot of poi's
# counts the number of poi's that possess a value other than 'NaN' for the 'poi_feature' in question
def count_poi_feature(poi_feature):
    return sum([1 if (data_dict[person][poi_feature] != 'NaN'
                      and data_dict[person]['poi'] == True)
                else 0 for person in data_dict])


### Maybe individuals with director fees are persons of interest
# Directors refer to senior employees appointed by shareholders to help run a company
# Directors fees are paid for attending board meetings
# However, 0 of the individuals with directors fees were found to be persons of interest.
count_poi_feature('director_fees')

### gives a list of features in which for a given feature, the difference between \
# 1) individuals with a value for that feature AND  \
# 2) individuals with a value for that feature and are also poi's \
# is less than 50
# the result: ['deferral_payments', 'loan_advances', 'deferred_income']
# In one way, this eliminates obvious features where everybody, poi or not, possesses such as 'salary' or 'bonus'
# This isn't saying that those features are unimportant, this is merely looking for features restricted to poi's
features_of_interest = []
for name in data_dict:
    for feature in data_dict[name]:
        if feature not in features_of_interest:
            list = []
            for person in data_dict:
                if data_dict[person][feature] != 'NaN':
                    list.append(person)
            if len(list) - count_poi_feature(feature) <= 50 and count_poi_feature(feature) != 0:
                features_of_interest.append(feature)

### Task 2: Remove outliers

# look for outliers with <= 1 features total
# ['LOCKHART EUGENE E']
# This data point has no values for any feature, safe to remove as it is an outlier
outlier_names = []
for name in data_dict:
    if name not in outlier_names:
        count = 0
        for feature in data_dict[name]:
            # ignore poi status
            if data_dict[name][feature] != 'NaN' and feature != 'poi':
                count += 1
        if count < 1:
            outlier_names.append(name)

outliers = ['THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E', 'TOTAL']
# Total is an outlier because it sums all the financial data for each feature
# The Travel Agency in the Park is an outlier because it only represents payments towards
# business-related travel to said place, definitely not an individual.
for outlier in outliers:
    data_dict.pop(outlier, 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# returns the fraction of total messages sent to and from that person to and from POI

my_dataset = data_dict


def frac(poi_messages, all_messages):
    if all_messages == 'NaN' or poi_messages == 'NaN' or all_messages == 0:
        return 0.
    else:
        fraction = float(float(poi_messages) / float(all_messages))
        return fraction


for name in my_dataset:
    person = my_dataset[name]

    from_poi_to_this_person = person['from_poi_to_this_person']
    to_messages = person['to_messages']
    fraction_from_poi = frac(from_poi_to_this_person, to_messages)
    person['fraction_from_poi'] = fraction_from_poi

    from_this_person_to_poi = person['from_this_person_to_poi']
    from_messages = person['from_messages']
    fraction_to_poi = frac(from_this_person_to_poi, from_messages)
    person['fraction_to_poi'] = fraction_to_poi

features_list = features_list + ['fraction_to_poi', 'fraction_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


pipeline = make_pipeline(MinMaxScaler(),
                    SelectKBest(),
                    #RandomizedPCA(),
                    #DecisionTreeClassifier())
                    #KNeighborsClassifier())
                    RandomForestClassifier())


# try out several different algorithms: DecisionTreeClassifier(), KNeighborsClassifier(), \
# and RandomForestClassifier()
params = {
    #'randomizedpca__n_components': [2, 4, 6, 8],
    #'randomizedpca__whiten': [True],
    'selectkbest__k': [5],
    'selectkbest__score_func': [chi2],
    #'decisiontreeclassifier__min_samples_split': [2, 10, 20, 30],
    #'decisiontreeclassifier__random_state': [42],
    #'kneighborsclassifier__n_neighbors': [1, 5, 10, 20],
    #'kneighborsclassifier__leaf_size': [1, 10, 25, 50],
    #'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'randomforestclassifier__n_estimators': [50],
    # n_estimators also included 5 and 25 but have been ommitted to shorten computing time
    'randomforestclassifier__random_state': [42]
}

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import train_test_split

# split data into training and testing datasets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, random_state=42)

# use StratifiedShuffleSplit for cross-validation
# random_state for reproducibility
cv = StratifiedShuffleSplit(labels_train,
                             n_iter = 20,
                             test_size = 0.3,
                             random_state=42)

# make estimator with GridSearchCV:
clf = GridSearchCV(pipeline,
                   param_grid=params,
                   cv=cv)

# fit model to clf
clf.fit(features_train, labels_train)

# For this Pipeline:
pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                       ('selectkbest', SelectKBest()),
                        ('randomforestclassifier', RandomForestClassifier())])
cv = StratifiedShuffleSplit(labels,n_iter = 50,test_size = 0.3, random_state = 42)
a_grid_search = GridSearchCV(pipe, param_grid = params,cv = cv, scoring = 'recall')
a_grid_search.fit(features,labels)

# pick a winner
best_clf = a_grid_search.best_estimator_ # This is the best classifier
print best_clf

# Gives a boolean list of indices of chosen ('True') features under selectkbest
pipeline.fit(features_train, labels_train)
selected_features = pipeline.named_steps['selectkbest'].get_support()

# Gives a list of chosen features
# ['bonus', 'deferred_income', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'salary',
# 'shared_receipt_with_poi', 'total_payments', 'total_stock_value', 'fraction_to_poi']
selected_features_names = [features_list[i + 1]
                           for i in pipeline.named_steps['selectkbest'].get_support(indices=True)]
#print selected_features_names

# Ranks feature importances in descending order, use 'decisiontreeclassifier' or 'randomforestclassifier'
# and its parameters in pipeline
#importances = pipeline.named_steps['randomforestclassifier'].feature_importances_
#indices = np.argsort(importances)[::-1]
# prints feature importances
#for i in range(len(selected_features_names)):
    #print (i+1,selected_features_names[indices[i]],importances[indices[i]])


# Gives feature scores (chi2 and p values) for the selected features (k=5)
# [quote = "Myles, post:3, topic:160463']
min_max_scaler = preprocessing.MinMaxScaler()
features_scaled = min_max_scaler.fit_transform(features)
k_selector = SelectKBest(chi2, k=5)
k_selector.fit_transform(features_scaled, labels)
feature_scores = ['%.2f' % elem for elem in k_selector.scores_]
feature_scores_pvalues = ['%.3f' % elem for elem in k_selector.pvalues_]
features_selected_tuple = [(features_list[i+1], feature_scores[i], feature_scores_pvalues[i])
                           for i in k_selector.get_support(indices=True)]
features_selected_tuple = sorted(features_selected_tuple,
                                 key = lambda feature: float(feature[1]),
                                 reverse = True)
#print features_selected_tuple
# [/quote]



# Evaluate classifier model
pred = clf.predict(features_test)
evaluation = classification_report(labels_test, pred, target_names = ['Not POI', 'POI'])
# print evaluation

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(best_clf, my_dataset, features_list)
