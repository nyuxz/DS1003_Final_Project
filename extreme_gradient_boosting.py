import sys
import os
#os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')
#sys.path.append('/Users/Sean/Desktop/DS1003_Final_Project')
#from importlib import reload
#reload(util)
import xgboost as xgb
import util
from util import regression_loss
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np 



def run_model(read_data_datapath, save_model_path):

	# read data
	x_train, x_test, y_train, y_test = util.prepare_train_test_set(read_data_datapath)

	# choose model
	clf = xgb.XGBRegressor(seed = 2017)

	# grid search for the best fit parameters
	param_grid = {

		'gamma': [0.0, 0.2, 0.4], # Minimum loss reduction required to make a further partition on a leaf node of the tree
		'max_depth': [3, 5, 7, 10], # in place of max_leaf_nodes
		'min_child_weight': [0.1, 1, 2], # Minimum sum of instance weight(hessian) needed in a child, in the place of min_child_leaf
		'n_estimators': [1000], # Number of boosted trees to fit
		'reg_alpha': [0.1, 0.5, 1.0], # L1 regularization term on weights
		'reg_lambda': [0.1, 0.5, 1.0] # L2 regularization term on weights

	}
	CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2)

	#CV_clf.fit(x_train[1:100,:], y_train[1:100])
	CV_clf.fit(x_train, y_train)


	# save model to pickle
	pickle.dump(CV_clf, open(save_model_path, "wb"))
	print('The best parameters are: \n %s' %CV_clf.best_params_)


	# run model and return loss
	#train_loss, test_loss = util.quick_test_model(x_train[1:100,:], x_test[1:100,:], y_train[1:100], y_test[1:100], CV_clf, regression_loss)
	train_loss, test_loss = util.quick_test_model(x_train, x_test, y_train, y_test, CV_clf, regression_loss)
	print("Train loss is %s, \n Test loss is %s  " % (train_loss, test_loss))


def show_result(read_model_path):
	# load model from file
	model = pickle.load(open(read_model_path, "rb"))
	model_result = model.cv_results_
	best_score = np.sqrt(-model.best_score_)
	print('The best parameters are: %s' %model.best_params_)
	print('The best RMSE is: %.3f' %best_score)

if __name__ == '__main__':

	run_model('./data/encoded_entire.pkl', "./model/xgb_entire_model.pkl")
	run_model('./data/encoded_others.pkl', "./model/xgb_others_model.pkl")

	show_result("./model/xgb_entire_model.pkl")
	show_result("./model/xgb_others_model.pkl")

