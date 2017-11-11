# coding: utf-8

# For python2.7
from __future__ import print_function

#Using Pandas python library 
import pandas as pd
import numpy as np
#labelEncoding to transform categorical values to Numerical values
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedShuffleSplit

# select rows in pandas frame by a list of index
from sklearn.utils import safe_indexing

def load_data(dataFile):
    print('Loading data from ' + dataFile)
    data = pd.read_csv(dataFile)

    #are there any missing values?
    print("Is there any missing values in raw data? ", end='')
    print(data.isnull().values.any())

    data['education'] = data.education.replace('unknown',np.nan)
    data['housing'] = data.housing.replace('unknown',np.nan)
    data['marital'] = data.marital.replace('unknown',np.nan)
    data['job'] = data.job.replace('unknown',np.nan)
    data['loan'] = data.loan.replace('unknown',np.nan)

    data = data.fillna(data['education'].value_counts().index[0])
    data = data.fillna(data['housing'].value_counts().index[0])
    data = data.fillna(data['marital'].value_counts().index[0])
    data = data.fillna(data['job'].value_counts().index[0])
    data = data.fillna(data['loan'].value_counts().index[0])

    # Create a label (category) encoder object
    le = preprocessing.LabelEncoder()
    data["job"] = le.fit_transform(data["job"])
    data["marital"] = le.fit_transform(data["marital"])
    data["education"] = le.fit_transform(data["education"])
    data["default"] = le.fit_transform(data["default"])
    data["housing"] = le.fit_transform(data["housing"])
    data["loan"] = le.fit_transform(data["loan"])
    data["contact"] = le.fit_transform(data["contact"])
    data["month"] = le.fit_transform(data["month"])
    data["day_of_week"] = le.fit_transform(data["day_of_week"])
    data["poutcome"] = le.fit_transform(data["poutcome"])
    data["y"] = le.fit_transform(data["y"])

    #are there any null values?
    data.isnull().values.any()
    print("Is there any missing values before feed into model? ", end='')
    print(data.isnull().values.any())

    for feature_name in ("pdays", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"):
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        data[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)

    x = data.drop("y", axis=1)
    x = x.drop("id", axis=1)
    y = data["y"]
    return x, y

'''
CrossValidation with stratified sampling cross-validation and example weights list.
'''
def my_cross_validate(model, x, y, cv, scoring, weight_list):
    print('\n************\nTraining on 5 folds cross-validation...')
    scores = {}
    for key in scoring:
        scores['train_' + key] = []
        scores['test_' + key] = []
    for train_index, test_index in cv.split(x, y):
        x_subset_train = safe_indexing(x, train_index)
        y_subset_train = safe_indexing(y, train_index)
        weight_list_subset = safe_indexing(weight_list, train_index)
        x_subset_test = safe_indexing(x, test_index)
        y_subset_test = safe_indexing(y, test_index)
        model.fit(x_subset_train, y_subset_train, weight_list_subset)

        # Get the training error of each fold.
        raw_predict = model.predict(x_subset_train)
        predicts = [round(value) for value in raw_predict]
        for key in scoring:
            if scoring[key] == 'accuracy':
                scores['train_' + key].append(accuracy_score(y_subset_train, predicts))
            else:
                scores['train_' + key].append(matthews_corrcoef(y_subset_train, predicts))

        # Get the test error of each fold.
        raw_predict = model.predict(x_subset_test)
        predicts = [round(value) for value in raw_predict]
        for key in scoring:
            if scoring[key] == 'accuracy':
                scores['test_' + key].append(accuracy_score(y_subset_test, predicts))
            else:
                scores['test_' + key].append(matthews_corrcoef(y_subset_test, predicts))
    for key in scores:
        scores[key] = np.array(scores[key])
    return scores


def print_scores(scores):
    print("Training accuracy: %0.2f (+/- %0.2f) %s" % (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2, scores['train_accuracy']))
    print("Training MCC: %0.5f (+/- %0.2f) %s" % (scores['train_MCC'].mean(), scores['train_MCC'].std() * 2, scores['train_MCC']))
    print("Test accuracy: %0.2f (+/- %0.2f) %s" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2, scores['test_accuracy']))
    print("Test MCC: %0.5f (+/- %0.2f) %s" % (scores['test_MCC'].mean(), scores['test_MCC'].std() * 2, scores['test_MCC']))

def pritn_param_score(weight, scores):
    print("Weights: ", weight)
    print("Training MCC: %0.5f (+/- %0.2f) %s" % (scores['train_MCC'].mean(), scores['train_MCC'].std() * 2, scores['train_MCC']))
    print("Test MCC: %0.5f (+/- %0.2f) %s" % (scores['test_MCC'].mean(), scores['test_MCC'].std() * 2, scores['test_MCC']))

'''
Assign each training example a weight by their category.
'''
def assign_weight(y, weight):
    weight_list = []
    for i in (y):
        if i == 0:
            weight_list.append(weight['0'])
        if i == 1:
            weight_list.append(weight['1'])
    return weight_list

def feature_importance_rank(model, x):
    rank = []
    # [::-1] means reverse the array. After reverse, the array arrange from big to small.
    rankIdx = np.argsort(model.feature_importances_)[::-1]
    for idx in rankIdx:
        rank.append(list(x)[idx])
    return rank

'''
Train the model on the whole training data,
make predictions on the test data,
generate the submission file 'output.txt'.
'''
def make_submission(model, param):
    print('\n****************\nBegin making submission file.')
    x, y = load_data('train.csv')
    print('Training the model on whole training data set...', end='')
    model.fit(x, y, assign_weight(y, param['class_weight']))
    print('Finished.')
    print(feature_importance_rank(model, x))
    x, y = load_data('test.csv')
    print('Make prediction on the test data set...', end='')
    predict = model.predict(x)
    print('Finished.')
    
    print('Writing the result into file \'output_' + param['name'] + '.csv\'...', end='')
    test = pd.read_csv('test.csv')
    test['prediction'] = predict
    predictions = [round(value) for value in test['prediction']]

    df2 = test[['id','prediction']]
    df2.to_csv('output_' + param['name'] + '.csv', index=False)
    print('Finished.')

def model1_with_parameter():
    param = {
        'name':'RF',
        'class_weight':{'0':1, '1':4},  # The weight of category.
        'n_estimators':250,
        'criterion':'gini', 
        'max_depth':None, 
        'min_samples_split':25, 
        'min_samples_leaf':1,
        'min_weight_fraction_leaf':0.0, 
        'max_features':'sqrt',
        'max_leaf_nodes':None,
        'bootstrap':True,
        'oob_score':False, 
        'n_jobs': -1, 
        'random_state':0, 
        'verbose':0
    }
    model = RandomForestClassifier(n_estimators=param['n_estimators'], criterion=param['criterion'], max_depth=param['max_depth'],
                               min_samples_split=param['min_samples_split'], min_samples_leaf=param['min_samples_leaf'],
                               min_weight_fraction_leaf=param['min_weight_fraction_leaf'], max_features=param['max_features'], 
                               max_leaf_nodes=param['max_leaf_nodes'], bootstrap=param['bootstrap'], oob_score=param['oob_score'],
                               n_jobs=param['n_jobs'], random_state=param['random_state'], verbose=param['verbose'])
    return model, param

def model2_with_parameter():
    param = {
        'name':'gdbt',
        'class_weight':{'0':1, '1':3},  # The weight of category.
        'loss':'deviance', 
        'learning_rate':0.1, 
        'n_estimators':300, 
        'subsample':1.0, 
        'criterion':'friedman_mse', 
        'min_samples_split':40, 
        'min_samples_leaf':20, 
        'min_weight_fraction_leaf':0.0, 
        'max_depth':3, 
        'min_impurity_decrease':0.0, 
        'min_impurity_split':None, 
        'init':None, 
        'random_state':5, 
        'max_features':None, 
        'verbose':0, 
        'max_leaf_nodes':None, 
        'warm_start':False,
        'presort':'auto'
    }
    model = GradientBoostingClassifier(loss=param['loss'], learning_rate=param['learning_rate'], n_estimators=param['n_estimators'],
                                subsample=param['subsample'], criterion=param['criterion'], min_samples_split=param['min_samples_split'], 
                                min_samples_leaf=param['min_samples_leaf'], min_weight_fraction_leaf=param['min_weight_fraction_leaf'], 
                                max_depth=param['max_depth'], min_impurity_decrease=param['min_impurity_decrease'], 
                                min_impurity_split=param['min_impurity_split'], init=param['init'], random_state=param['random_state'], 
                                max_features=param['max_features'], verbose=param['verbose'], max_leaf_nodes=param['max_leaf_nodes'], 
                                warm_start=param['warm_start'], presort=param['presort'])
    return model, param

def model_tuning():
    # Load training data set.
    x, y = load_data('train.csv') #[30891 rows x 21 columns]

    # Define the model.
    model, param = model1_with_parameter()

    # Evaluation criterias of our model. One is Accuracy, another is MCC.
    scorer = metrics.make_scorer(matthews_corrcoef)
    scoring = {"accuracy":"accuracy", "MCC":scorer}

    # Define the Cross-validation partition.
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # Perform the cross_validation.
    scores = my_cross_validate(model, x, y, cv, scoring=scoring, weight_list=assign_weight(y, param['class_weight']))
    #print_scores(scores)
    pritn_param_score(param, scores)

    '''
    For paramters search.
    '''
    # for i in xrange(2, 10):
    #     weight['1'] = i
    #     scores = my_cross_validate(model, x, y, cv, scoring=scoring, weight_list=assign_weight(y, weight))
    #     pritn_param_score(weight, scores)

    # Generate predictions on test data to submit online.
    make_submission(model, param)

def ensumble_vote():
    pass

if __name__=='__main__':
    model_tuning()
    #ensumble_vote()
