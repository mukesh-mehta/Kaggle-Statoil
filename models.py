from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, concatenate, Input, Flatten, Activation, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16
from keras.initializers import glorot_uniform
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from processing import pre_processing_1
import keras

X_train, X_angle,target_train, X_test, X_test_angle=pre_processing_1()
def Vgg16Model():
	input_2 = Input(shape=[1], name="angle")
	angle_layer = Dense(1, )(input_2)
	base_model = VGG16(weights='imagenet', include_top=False, 
	             input_shape=X_train.shape[1:], classes=1)
	x = base_model.get_layer('block5_pool').output

	x = GlobalMaxPooling2D()(x)
	merge_one = concatenate([x, angle_layer])
	merge_one = Dense(512, activation='relu', name='fc2', kernel_initializer = glorot_uniform(seed=0))(merge_one)
	merge_one = Dropout(0.3)(merge_one)
	merge_one = Dense(512, activation='relu', name='fc3', kernel_initializer = glorot_uniform(seed=0))(merge_one)
	merge_one = Dropout(0.3)(merge_one)

	predictions = Dense(1, activation='sigmoid')(merge_one)

	model = Model(input=[base_model.input, input_2], output=predictions)

	#adam =Adam(lr=1e-4) 
	sgd=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

def Vgg19Model():
	input_2 = Input(shape=[1], name="angle")
	angle_layer = Dense(1, )(input_2)
	base_model = VGG16(weights='imagenet', include_top=False, 
	             input_shape=X_train.shape[1:], classes=1)
	x = base_model.get_layer('block5_pool').output

	x = GlobalMaxPooling2D()(x)
	merge_one = concatenate([x, angle_layer])
	merge_one = Dense(512, activation='relu', name='fc2', kernel_initializer = glorot_uniform(seed=0))(merge_one)
	merge_one = Dropout(0.3)(merge_one)
	merge_one = Dense(512, activation='relu', name='fc3', kernel_initializer = glorot_uniform(seed=0))(merge_one)
	merge_one = Dropout(0.3)(merge_one)

	predictions = Dense(1, activation='sigmoid')(merge_one)

	model = Model(input=[base_model.input, input_2], output=predictions)

	#adam =Adam(lr=1e-4) 
	sgd=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

def Vgg16_mobilenetModel():
	input_2 = Input(shape=[1], name="angle")
	#changed dense1 to dense 75
	angle_layer = Dense(1, )(input_2)
	base_model = VGG16(weights='imagenet', include_top=False,input_shape=X_train.shape[1:], classes=1)
	x = base_model.get_layer('block5_pool').output
	x = GlobalMaxPooling2D()(x)
	base_model2 = keras.applications.mobilenet.MobileNet(weights=None, alpha=0.9,input_tensor = base_model.input,include_top=False, input_shape=X_train.shape[1:])

	x2 = base_model2.output
	x2 = GlobalAveragePooling2D()(x2)

	merge_one = concatenate([x, x2, angle_layer])

	merge_one = Dropout(0.6)(merge_one)
	predictions = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(merge_one)

	model = Model(input=[base_model.input, input_2], output=predictions)

	sgd = Adam(lr=1e-4) #SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model

def basicModel():
    model=Sequential()
    # CNN 1
    model.add(Conv2D(8, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))
    # CNN 2
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # CNN 3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    #CNN 5
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    # You must flatten the data for the dense layers
    model.add(Flatten())
    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # Output 
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

