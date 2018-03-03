"""
It takes 2 input files, train file and test file and returns
X_train, X_angle
X_test, X_test_angle
"""
import pandas as pd
import numpy as np

train_file="data/train.json"
test_file="data/test.json"

def pre_processing_1():
	train=pd.read_json(train_file)
	test=pd.read_json(test_file)

	target_train=train['is_iceberg']

	train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')
	X_angle=train['inc_angle'].fillna(method='pad')

	test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
	X_test_angle=test['inc_angle']

	#Generate the training data
	X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
	X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
	#X_band_3=(X_band_1+X_band_2)/2
	X_band_3=np.fabs(np.subtract(X_band_1,X_band_2))
	X_band_4=np.maximum(X_band_1,X_band_2)
	X_band_5=np.minimum(X_band_1,X_band_2)
	#X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
	X_train = np.concatenate([X_band_3[:, :, :, np.newaxis],X_band_4[:, :, :, np.newaxis],X_band_5[:, :, :, np.newaxis]], axis=-1)



	X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
	X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
	#X_band_test_3=(X_band_test_1+X_band_test_2)/2
	X_band_test_3=np.fabs(np.subtract(X_band_test_1,X_band_test_2))
	X_band_test_4=np.maximum(X_band_test_1,X_band_test_2)
	X_band_test_5=np.minimum(X_band_test_1,X_band_test_2)
	#X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
	X_test = np.concatenate([ X_band_test_3[:, :, :, np.newaxis], X_band_test_4[:, :, :, np.newaxis],X_band_test_5[:, :, :, np.newaxis]],axis=-1)
	return X_train, X_angle,target_train, X_test, X_test_angle

def pre_processing_2():
	train = pd.read_json(train_file)
	test=pd.read_json(test_file)

	X_band1=np.array([np.array(band).astype(np.float32).reshape(75, 75,1) for band in train["band_1"]])
	X_band2=np.array([np.array(band).astype(np.float32).reshape(75, 75,1) for band in train["band_2"]])
	X_train=np.concatenate([X_band1[:, :, :, np.newaxis], X_band2[:, :, :, np.newaxis]],axis=-1)

	#test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
	train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')#We have only 133 NAs.
	train['inc_angle']=train['inc_angle'].fillna(method='pad')
	X_angle=train['inc_angle']

	Y_train=train['is_iceberg']

	X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
	X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
	test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
	X_test_angle=test['inc_angle']

	return X_band1, X_band2,X_angle, Y_train, X_band_test_1, X_band_test_2, X_test_angle