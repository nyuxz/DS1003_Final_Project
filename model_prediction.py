import pickle
import sys
import os
import util
import pandas as pd
from util import regression_loss
#os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')
#sys.path.append('/Users/Sean/Desktop/DS1003_Final_Project')


# entire
datapath = './data/encoded_entire.pkl'
pkl_file = open(datapath, 'rb')
dataset = pickle.load(pkl_file)
read_model_path = "./model/xgb_final/xgb_entire_model_final.pkl"
model_entire = pickle.load(open(read_model_path, "rb"))
x_train_entire, x_test_entire, y_train_entire, y_test_entire = util.prepare_train_test_set('./data/encoded_entire.pkl', 0.001)
prediction_entire = model_entire.predict(x_test_entire)



# private
datapath = './data/encoded_private.pkl'
pkl_file = open(datapath, 'rb')
dataset = pickle.load(pkl_file)
read_model_path = "./model/xgb_final/xgb_private_model_final.pkl"
model_private = pickle.load(open(read_model_path, "rb"))
x_train_private, x_test_private, y_train_private, y_test_private = util.prepare_train_test_set('./data/encoded_private.pkl', 0.001)
prediction_private = model_private.predict(x_test_private)



prediction_entire = pd.DataFrame({'y_test_entire' : list(y_test_entire), 'prediction_entire' : list(prediction_entire)})
prediction_entire.to_csv('./model/prediction_entire.csv')


prediction_private = pd.DataFrame({'y_test_private': list(y_test_private), 'prediction_private': list(prediction_private)})
prediction_private.to_csv('./model/prediction_private.csv')


#-------------------------------------------------------------------------------

# entire
from sklearn.model_selection import train_test_split
label = 'price'
datapath = './data/data_entire.pkl'
pkl_file = open(datapath, 'rb')
dataset = pickle.load(pkl_file)
dataset = pd.DataFrame(dataset)
x = dataset.drop([label], 1)
y = dataset[label]
x_train_entire, x_test_entire, y_train_entire, y_test_entire = train_test_split(x, y, test_size=0.2,
                                              random_state=0)


# private
from sklearn.model_selection import train_test_split
label = 'price'
datapath = './data/data_private.pkl'
pkl_file = open(datapath, 'rb')
dataset = pickle.load(pkl_file)
dataset = pd.DataFrame(dataset)
x = dataset.drop([label], 1)
y = dataset[label]
x_train_private, x_test_private, y_train_private, y_test_private = train_test_split(x, y, test_size=0.2,
                                              random_state=0)





x_test_entire = pd.DataFrame(x_test_entire)
x_test_entire.to_csv('./model/x_test_entire.csv')
