from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.models import model_from_json
from keras.utils import np_utils, generic_utils
import numpy as np
import sys
import cPickle as pickle
import argparse
def extract_test_seq(file_name,file_tag_name,tag_size):
	
	X = []
	Y = []
	file_fin = open(file_name)
	tag_fin = open(file_tag_name)
	counter = 0
	for line in tag_fin.readlines():
		line_body = file_fin.readline()
		coco = line.split()
		if len(coco) == 0:
			coco = [tag_size + 100]
		coco2 = line_body.split()
		temp_y = map(int,coco)
		temp_x = map(int,coco2)
		X.append(temp_x)
		Y.append(temp_y)
		counter += 1
	file_fin.close()
	tag_fin.close()
	return(X,Y)

def extractseq(file_name,file_tag_name,tag_size):
	
	X = []
	Y = []
	file_fin = open(file_name)
	tag_fin = open(file_tag_name)
	counter = 0
	for line in tag_fin.readlines():
		#print counter , '\n'
		line_body = file_fin.readline()
		coco = line.split()
		if len(coco) == 0:
			continue
		coco2 = line_body.split()
		temp_y = map(int,coco)
		temp_x = map(int,coco2)
		X.append(temp_x)
		Y.append(temp_y)
		counter += 1
	file_fin.close()
	tag_fin.close()
	return(X,Y)

def getmaxlen(X):
	count = 0
	for coco in X:
		if len(coco) > count:
			count = len(coco)
	return count

def buildlabel(Y,tag_num):
	coco = []
	counter = 0
	for data in Y:
		#print counter
		temp = [0]*tag_num
		length = len(data)
		for tag in data:
			temp[tag-1] = float(1)/float(length)
			#temp[tag-1] = 0
			#temp[tag-1] = 1/length
		coco.append(temp)
		counter += 1
	coco = np.array(coco)
	return coco

if __name__ == '__main__':
	
	datapathin = './splitdata/'
	vectorpathin = './splitvector/'
	target = 'interspeech'
	source = 'stack'
	file_name_train = vectorpathin+target+'.train.body.vector'
	file_name_test = vectorpathin+target+'.test.body.vector'
	file_name_train_tag = vectorpathin+target+'.train.tag.vector'
	file_name_test_tag = vectorpathin+target+'.test.tag.vector'
	file_dic_name = datapathin+'All.dic.body'
	pathweight = './weight/'
	weightname = target+'.pretrain.weight6_16_embed15026'
	resultfile = './result/'+target+'.result1108try'

	parser = argparse.ArgumentParser()
	parser.add_argument("-weight",type=str)
	parser.add_argument("-output_weight",type=str)
	args = parser.parse_args()
	outputweight = args.output_weight

	fin = open(file_dic_name,'r')
	coco = fin.readlines()
	fin.close()
	#parameters
	print '=================='
	print 'Source Domain :',source
	print 'Train On Target Domain :',target
	print '=================='
	batch_size = 64
	maxlen = 300
	max_features = len(coco)+1
	print 'max_features: ',max_features
	embedding_dim = 300 # 1108 for mycorpus
	d_lstm = 128
	d_dense = 512
	d_output = 1500
	d_lowdim = 50
	epoch = 1
	print 'Batch_Size :',batch_size
	print 'Maxlen :',maxlen
	print 'd_lstm :',d_lstm
	print 'd_dense :',d_dense
	print 'd_lowdim:',d_lowdim
	print 'd_output :',d_output
	print 'epoch :',epoch
	print '=================='
	#load data
	print 'Loading Test Data'
	(X_test,Y_test) = extractseq(file_name_test,file_name_test_tag,d_output)
	print 'Test Data Number :',len(Y_test)
	#sys.exit(1)
	Y_test = buildlabel(Y_test,d_output)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	print '=================='
	print 'Loading Train Data'
	(X_train,Y_train) = extractseq(file_name_train,file_name_train_tag,d_output)
	print 'Train Data Number :',len(Y_train)
	X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	Y_train = buildlabel(Y_train,d_output)
	print '=================='
	#build NN
	print 'Building model'
	model = Sequential()
	model.add(Embedding(max_features, embedding_dim, input_length=maxlen, init='glorot_uniform',trainable = True))
	model.add(LSTM(output_dim = d_lstm, return_sequences=False, input_shape=(maxlen, embedding_dim), init='glorot_uniform', inner_init='orthogonal', inner_activation='hard_sigmoid',trainable = True))
	model.add(Dense(d_dense, init='glorot_uniform',trainable = True))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(d_dense, init='glorot_uniform',trainable = True))
	model.add(Activation('relu'))
#	model.add(Dense(d_lowdim, init='glorot_uniform',trainable = True))
#	model.add(Activation('relu'))
	print 'Using Pretrain Weight :', args.weight
	#coco = pickle.load(open(pathweight+weightname));
	coco = pickle.load(open(args.weight));
	cocoweight = [coco[i] for i in range(17)]
	model.set_weights(cocoweight)
	model.add(Dense(d_output, init = 'glorot_uniform',trainable = True))
	model.add(Activation('softmax'))
	#model.set_weights(coco)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	rmsprop = RMSprop(lr=0.0008, rho=0.9, epsilon=1e-06)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
	#Train
	print '=================='
	for time in range(8):
		#break
		print time
		model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch,
		validation_data=(X_test, Y_test))
		coco = model.get_weights()
		fileout = open(pathweight+outputweight+str(time),'w+')
		pickle.dump(coco,fileout)
	#Test

	#load test data
	print 'Loading Test Data'
	(X_test,Y_test) = extract_test_seq(file_name_test,file_name_test_tag,d_output)
	print 'Test num:',len(Y_test)
	#sys.exit(1)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	coco_chung = model.predict(X_test)
	fout = open(resultfile,'w+')
	for item in coco_chung:
		coco = map(str,item)
		coco = ' '.join(coco)
		fout.write(coco+'\n')
	fout.close()

