from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from utils import limit_mem, gen_flow_for_two_inputs, get_callbacks
from models import Vgg16Model, Vgg19Model, Vgg16_inceptionModel, basicModel

def train_vgg_mobile(X_train,X_angle,target_train,X_test,X_test_angle,K):
	print("Running vgg and mobilenet")
	folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
	y_test_pred_log = 0
	y_train_pred_log=0
	y_valid_pred_log = 0.0*target_train
	for j, (train_idx, test_idx) in enumerate(folds):
		print('\n===================FOLD=',j+1)
		limit_mem()
		X_train_cv = X_train[train_idx]
		y_train_cv = target_train[train_idx]
		X_holdout = X_train[test_idx]
		Y_holdout= target_train[test_idx]

		#Angle
		X_angle_cv=X_angle[train_idx]
		X_angle_hold=X_angle[test_idx]

		#define file path and get callbacks
		file_path = "%s_aug_model_weights.hdf5"%j
		callbacks = get_callbacks(filepath=file_path, patience=10)
		gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
		galaxyModel= Vgg16_inceptionModel()
		galaxyModel.fit_generator(
		        gen_flow,
		        steps_per_epoch=24,
		        epochs=100,
		        shuffle=True,
		        verbose=1,
		        validation_data=([X_holdout,X_angle_hold], Y_holdout),
		        callbacks=callbacks)

		#Getting the Best Model
		galaxyModel.load_weights(filepath=file_path)
		#Getting Training Score
		score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
		print('Train loss:', score[0])
		print('Train accuracy:', score[1])
		#Getting Test Score
		score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		#Getting validation Score.
		pred_valid=galaxyModel.predict([X_holdout,X_angle_hold])
		y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

		#Getting Test Scores
		temp_test=galaxyModel.predict([X_test, X_test_angle])
		y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

		#Getting Train Scores
		temp_train=galaxyModel.predict([X_train, X_angle])
		y_train_pred_log+=temp_train.reshape(temp_train.shape[0])

	y_test_pred_log=y_test_pred_log/K
	y_train_pred_log=y_train_pred_log/K

	print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
	print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
	return y_test_pred_log

def train_vgg16(X_train,X_angle,target_train,X_test,X_test_angle,K):
	print("Running vgg16")
	folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
	y_test_pred_log = 0
	y_train_pred_log=0
	y_valid_pred_log = 0.0*target_train
	for j, (train_idx, test_idx) in enumerate(folds):
		print('\n===================FOLD=',j+1)
		limit_mem()
		X_train_cv = X_train[train_idx]
		y_train_cv = target_train[train_idx]
		X_holdout = X_train[test_idx]
		Y_holdout= target_train[test_idx]

		#Angle
		X_angle_cv=X_angle[train_idx]
		X_angle_hold=X_angle[test_idx]

		#define file path and get callbacks
		file_path = "%s_aug_model_weights.hdf5"%j
		callbacks = get_callbacks(filepath=file_path, patience=10)
		gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
		galaxyModel= Vgg16Model()
		galaxyModel.fit_generator(
		        gen_flow,
		        steps_per_epoch=24,
		        epochs=100,
		        shuffle=True,
		        verbose=1,
		        validation_data=([X_holdout,X_angle_hold], Y_holdout),
		        callbacks=callbacks)

		#Getting the Best Model
		galaxyModel.load_weights(filepath=file_path)
		#Getting Training Score
		score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
		print('Train loss:', score[0])
		print('Train accuracy:', score[1])
		#Getting Test Score
		score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		#Getting validation Score.
		pred_valid=galaxyModel.predict([X_holdout,X_angle_hold])
		y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

		#Getting Test Scores
		temp_test=galaxyModel.predict([X_test, X_test_angle])
		y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

		#Getting Train Scores
		temp_train=galaxyModel.predict([X_train, X_angle])
		y_train_pred_log+=temp_train.reshape(temp_train.shape[0])

	y_test_pred_log=y_test_pred_log/K
	y_train_pred_log=y_train_pred_log/K

	print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
	print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
	return y_test_pred_log

def train_vgg19(X_train,X_angle,target_train,X_test,X_test_angle,K):
	print("Running vgg19")
	folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
	y_test_pred_log = 0
	y_train_pred_log=0
	y_valid_pred_log = 0.0*target_train
	for j, (train_idx, test_idx) in enumerate(folds):
		print('\n===================FOLD=',j+1)
		limit_mem()
		X_train_cv = X_train[train_idx]
		y_train_cv = target_train[train_idx]
		X_holdout = X_train[test_idx]
		Y_holdout= target_train[test_idx]

		#Angle
		X_angle_cv=X_angle[train_idx]
		X_angle_hold=X_angle[test_idx]

		#define file path and get callbacks
		file_path = "%s_aug_model_weights.hdf5"%j
		callbacks = get_callbacks(filepath=file_path, patience=10)
		gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
		galaxyModel= Vgg19Model()
		galaxyModel.fit_generator(
		        gen_flow,
		        steps_per_epoch=24,
		        epochs=100,
		        shuffle=True,
		        verbose=1,
		        validation_data=([X_holdout,X_angle_hold], Y_holdout),
		        callbacks=callbacks)

		#Getting the Best Model
		galaxyModel.load_weights(filepath=file_path)
		#Getting Training Score
		score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
		print('Train loss:', score[0])
		print('Train accuracy:', score[1])
		#Getting Test Score
		score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		#Getting validation Score.
		pred_valid=galaxyModel.predict([X_holdout,X_angle_hold])
		y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

		#Getting Test Scores
		temp_test=galaxyModel.predict([X_test, X_test_angle])
		y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

		#Getting Train Scores
		temp_train=galaxyModel.predict([X_train, X_angle])
		y_train_pred_log+=temp_train.reshape(temp_train.shape[0])

	y_test_pred_log=y_test_pred_log/K
	y_train_pred_log=y_train_pred_log/K

	print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
	print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
	return y_test_pred_log

def train_basic(X_band1, X_band2,X_angle, Y_train, X_band_test_1, X_band_test_2, X_test_angle):
	print("Running Basic model")
	model=basicModel()
	model.fit([X_band1, X_band2, X_angle],Y_train,epochs=20,batch_size=32, validation_split=0.2)
	preds= model.predict_proba([X_band_test_1, X_band_test_2, X_test_angle], batch_size=32)
	return preds