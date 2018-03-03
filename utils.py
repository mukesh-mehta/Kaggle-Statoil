#utilities
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))
    print('gpu memory cleaned')

def get_callbacks(filepath, patience=10):
    #es = EarlyStopping('val_loss', patience=10, mode="min")
    es = EarlyStopping('val_loss', patience=10, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def gen_flow_for_two_inputs(X1, X2, y, horizontal_flip = True, vertical_flip = True,
							 width_shift_range = 0., height_shift_range = 0.,
							  channel_shift_range=0, zoom_range = 0.5, rotation_range = 10):

	gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0.5,
                         rotation_range = 10)
	genX1 = gen.flow(X1,y,  batch_size=64,seed=55)
	genX2 = gen.flow(X1,X2, batch_size=64,seed=55)
	while True:
		X1i = genX1.next()
		X2i = genX2.next()
		#Assert arrays are equal - this was for peace of mind, but slows down training
		#np.testing.assert_array_equal(X1i[0],X2i[0])
		yield [X1i[0], X2i[1]], X1i[1]

