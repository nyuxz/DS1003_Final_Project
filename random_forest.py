# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:52:09 2017

@author: siyang
"""

import sys
import os

# Change your path
#os.chdir('/Users/siyang/Downloads/DS1003_Final_Project-master/')
#sys.path.append('/Users/siyang/Downloads/DS1003_Final_Project-master/')

# Codes starts here

# Import the necessary modules and libraries
import util
from util import regression_loss
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
#import matplotlib.pyplot as plt
import pickle

def run_model(read_data_datapath, save_model_path):
    # read data
    x_cv, x_test, y_cv, y_test = util.prepare_train_test_set(read_data_datapath)
    # choose model
    clf = Pipeline([('clf',RandomForestRegressor(criterion='mse',random_state=0))])
    # grid search for the best fit parameters
    parameters = {
        'clf__n_estimators': [1000], #number of trees
        'clf__max_depth': (125, 100, 75, 50, 40, 30, 25, 10),
        'clf__min_samples_split': (2, 3, 4, 5, 6 ),
        'clf__min_samples_leaf': (1, 2, 3, 4, 5, 6),
        'clf__min_impurity_split': (1e-8, 1e-7, 1e-6, 1e-5)
    }
        
    CV_clf = GridSearchCV(estimator=clf, param_grid = parameters, cv=3, scoring='neg_mean_squared_error')
    
    CV_clf.fit(x_cv, y_cv)
    #CV_clf.fit(x_cv[1:100,:], y_cv[1:100])
    print ('Best score: %0.3f' % CV_clf.best_score_)
    # save model to pickle
    pickle.dump(CV_clf, open(save_model_path, "wb"))
    print ('Best parameters set are: \n %s' %  CV_clf.best_estimator_.get_params())


    # run model and return loss
    train_loss, test_loss = util.quick_test_model(x_cv, x_test, y_cv, y_test, CV_clf, regression_loss)
    #train_loss, test_loss = util.quick_test_model(x_cv[1:100,:], x_test[1:100,:], y_cv[1:100], y_test[1:100], CV_clf, regression_loss)
    print("Train loss is %s, \n Test loss is %s  " % (train_loss, test_loss))

def show_result(read_model_path):
    # load model from file
    model = pickle.load(open(read_model_path, "rb"))
    #model_result = model.cv_results_
    best_score = np.sqrt(-model.best_score_)
    print('The best parameters are: %s' %model.best_params_)
    print('The best RMSE is: %.3f' %best_score)
    
        
if __name__ == '__main__':
    run_model('./data/encoded_entire.pkl', "./model/forest_entire_model.pkl")
    run_model('./data/encoded_others.pkl', "./model/forest_others_model.pkl")

    show_result("./model/forest_entire_model.pkl")
    show_result("./model/forest_others_model.pkl")





# Feature importances

#importances = CV_clf.feature_importances_
#std = np.std([tree.feature_importances_ for tree in CV_clf.estimators_],
             #axis=0)
#indices = np.argsort(importances)[::-1]


# Print the feature ranking
#print("Feature ranking:")

#for f in range(x_train.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the results

#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(x_train.shape[1]), importances[indiches],
 #color="r", yerr=std[indices], align="center")
#plt.xticks(range(x_train.shape[1]), indices)
#plt.xlim([-1, x_train.shape[1]])
#plt.show()


# Visualization 
#import pydot  
#tree.export_graphviz(dtreg, out_file='tree.dot') #produces dot file
#dotfile = StringIO()
#tree.export_graphviz(dtreg, out_file=dotfile)
#pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png") 
   
