import sys
import os

# Change your path

os.chdir('/Users/siyang/Downloads/DS1003_Final_Project-master/')
sys.path.append('/Users/siyang/Downloads/DS1003_Final_Project-master/')

# Codes starts here

# Import the necessary modules and libraries
#from importlib import reload
#reload(util)
import util
from util import regression_loss
#from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import pickle
#import matplotlib.pyplot as plt

def run_model(read_data_datapath, save_model_path):
    # read data
    x_cv, x_test, y_cv, y_test = util.prepare_train_test_set(read_data_datapath)
    # choose model
    clf = Pipeline([('clf',DecisionTreeRegressor(criterion='mse',random_state=0))])
    #clf = DecisionTreeRegressor(criterion='mse',random_state=0)
    # grid search for the best fit parameters
    parameters = {
        'clf__max_depth': [125, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5],
        'clf__min_samples_split': [2, 3, 4 ,5, 6],
        'clf__min_samples_leaf': [1, 2, 3, 4 ,5, 6]  
    }
        
    CV_clf = GridSearchCV(estimator=clf, param_grid = parameters, cv=3, scoring='neg_mean_squared_error')

    #CV_clf.fit(x_cv[1:100,:], y_cv[1:100])
    CV_clf.fit(x_cv, y_cv)
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
    model_result = model.cv_results_
    best_score = np.sqrt(-model.best_score_)
    print('The best parameters are: %s' %model.best_params_)
    print('The best RMSE is: %.3f' %best_score)
    # visualizing purpose
    #rmse_list = np.sqrt(-model_result['mean_test_score'])
    
        
if __name__ == '__main__':
    run_model('./data/encoded_entire.pkl', "./model/tree_entire_model.pkl")
    run_model('./data/encoded_others.pkl', "./model/tree_others_model.pkl")

    show_result("./model/tree_entire_model.pkl")
    show_result("./model/tree_others_model.pkl")


# Predict

#y_pred = CV_clf.predict(x_test)


# Visualization 
#import pydot  
#tree.export_graphviz(dtreg, out_file='tree.dot') #produces dot file
#dotfile = StringIO()
#tree.export_graphviz(dtreg, out_file=dotfile)
#pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png") 
   
