import tensorflow.python.framework.dtypes
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import applications, optimizers
from random import sample
import numpy as np
import os



Y_true=17.19404
ns=10
bias_array = np.zeros( (ns, 2) )
data_path = "M:/machine learning missing data/simulation/p1000_Response0.5_cor0/"
fn_res = data_path + "resq.txt"
learning_rate=0.015

for j in range(ns):
	i=j+1
	fn_comp = data_path + str(i) + "_comp.txt"
	fn_miss = data_path + str(i) + "_miss.txt"
	fn_impt = data_path + str(i) + "_dl.txt"
	d_comp=np.loadtxt(fn_comp,delimiter=",")
	d_miss=np.loadtxt(fn_miss,delimiter=",")

	ncol=d_comp.shape[1]
	nrow=d_comp.shape[0]
	nt=nrow+d_miss.shape[0]
	Y=d_comp[:,0]
	X=d_comp[:,1:ncol]
	Xt=d_miss[:,1:ncol]
	Yt=d_miss[:,0]
	resp_inx=np.concatenate((np.repeat(1,nrow),np.repeat(0,d_miss.shape[0])))
	epochs=200
	batch_size=32
	#model = ResNet50Regression(X)
	#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
	#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
	#mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
	#model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.4, callbacks=[es, mc])
	#saved_model = tensorflow.keras.models.load_model('best_model.h5')
	model = Sequential()
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.4))
	#model.add(Dense(32, activation='elu'))
	#model.add(Dropout(0.4))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(8, activation="linear"))
	model.add(Dense(1, kernel_initializer="glorot_normal"))
	learning_rate=0.01
	opt = optimizers.Adam(lr=learning_rate)
	#model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
	#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
	#mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
	#model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.4, callbacks=[es, mc])
	#saved_model = load_model('best_model.h5')	
	#Yp=saved_model.predict(Xt)
	#bias_array[j,0]=(Yp.sum()+Y.sum())/nt - Y_true
	

	##DL model 2: FCN
	model = Sequential()
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(16, activation='linear'))
	model.add(Dense(1, kernel_initializer="glorot_normal"))
	learning_rate=0.01
	opt = optimizers.Adam(lr=learning_rate)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
	mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
	model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.4, callbacks=[es, mc])
	saved_model = load_model('best_model.h5')
	Yp=saved_model.predict(Xt)
	bias_array[j,1]=(Yp.sum()+Y.sum())/nt - Y_true
	
	

	#np.abs(Yp-Yt).mean()

print(np.mean(bias_array,axis=0))
print(np.std(bias_array,axis=0))
#np.savetxt(fn_res, bias_array)

