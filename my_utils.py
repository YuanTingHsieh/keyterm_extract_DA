import numpy as np
import math
from sklearn.preprocessing import normalize

'''keras'''
from keras.models import Sequential, Model
from keras.layers import Activation, Embedding, LSTM
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop

## my own library
from mylayer import *


def get_dict(path):
	print 'Geting dict from '+path
	counter = 0
	word_indices = {}
	indices_word = {}
	for line in open(path,'r').read().splitlines():
		word_indices.update({line:counter})
		indices_word.update({counter:line})
		counter += 1
	print 'Dict len is ',len(word_indices)
	print '=================='
	return word_indices, indices_word

def vectorize_label(path, top_k, word_indices):
	the_lines = open(path).read().lower().splitlines()
	totallines = len(the_lines)
	valid_index = np.zeros(totallines).astype(np.bool_)
	oov = np.zeros(totallines).astype(np.int32)
	y=np.zeros((totallines, top_k),dtype=np.float32)
	for i,lines in enumerate(the_lines):
		for j in lines.split():
			if j in word_indices:
				y[i, word_indices[j] ] =1
				valid_index[i] = 1
			else:
				oov[i] = oov[i]+1 
	y_norm = normalize(y, norm='l1')
	return y, y_norm, valid_index, oov

def printParams(args, max_features, maxlen,d_output):
	print 'Max Feature: ',max_features
	print 'Batch_Size :',args.batch_size
	print 'Max Length :',maxlen
	print 'd_lstm     :',args.lstm_dim
	print 'd_dense    :',args.dense_dim
	print 'd_output   :',d_output
	print 'd_embed    :',args.embed_dim
	print 'epochs     :',args.epochs
	print '=================='

def extractseq(file_name,file_tag_name):
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

def extractseqNaive(file_name,file_tag_name,max_features):
	Y = []
	file_fin = open(file_name)
	tag_fin = open(file_tag_name)
	counter = 0
	ind = []
	for line in tag_fin.readlines():
		#print counter , '\n'
		counter += 1
		coco = line.split()
		if len(coco) == 0:
			continue
		temp_y = map(int,coco)
		Y.append(temp_y)
		ind.append(counter)
	counter = 0
	count_row = 0
	X = np.zeros((len(ind),max_features), dtype=np.float32)
	print len(ind)
	print ind[len(ind)-1]
	for line_body in file_fin.readlines():
		counter += 1
		if counter in ind:
			num_x = map(int,line_body.split())
			for a in num_x:
				X[count_row,a-1] += 1
			count_row+=1
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
			#temp[tag-1] = 1
		coco.append(temp)
		counter += 1
	coco = np.array(coco)
	return coco

def myloadDataNaive(file_name, file_name_tag, d_output, max_features):
	print 'Loading Data Naive'
	(X,Y) = extractseqNaive(file_name, file_name_tag, max_features)
	print 'Data Number :',len(Y)
	Y = buildlabel(Y, d_output)
	print '=================='
	return (X, Y)

def myloadData(file_name, file_name_tag, d_output, maxlen):
	print 'Loading Data'
	(X,Y) = extractseq(file_name, file_name_tag)
	print 'Data Number :',len(Y)
	Y = buildlabel(Y, d_output)
	X = sequence.pad_sequences(X, maxlen=maxlen)
	print '=================='
	X = X.astype(np.float32)
	Y = Y.astype(np.float32)
	return (X, Y)

def Naive(max_features,args, d_output):
	d_dense = args.dense_dim
	print 'Building Naive model'
	model = Sequential()
	model.add(Dense(d_dense, init='glorot_uniform',input_shape= (max_features,)))
	model.add(Activation('relu'))
	model.add(Dense(d_dense, init='glorot_uniform'))
	model.add(Activation('relu'))
	model.add(Dense(d_output))
	model.add(Activation('softmax'))
	rmsprop = RMSprop(lr=0.0008, rho=0.9, epsilon=1e-06)
	#sgd = SGD(lr=0.1)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop,metrics=['accuracy'])
	print '=================='
	return model

# using binary cross-entro
def myLSTM2(max_features, maxlen, args, d_output):
	embedding_dim = args.embed_dim
	d_lstm = args.lstm_dim
	d_dense = args.dense_dim
	print 'Building model'
	model = Sequential()
	model.add(Embedding(max_features, embedding_dim, input_length=maxlen, init='glorot_uniform'))
	model.add(LSTM(output_dim = d_lstm, return_sequences=False, input_shape=(maxlen, embedding_dim), init='glorot_uniform', inner_init='orthogonal', inner_activation='hard_sigmoid'))
	model.add(Dense(d_dense, init='glorot_uniform'))
	model.add(Activation('relu'))
#	model.add(Dropout(0.1))
	model.add(Dense(d_dense, init='glorot_uniform'))
	model.add(Activation('relu'))
	model.add(Dense(d_output))
	model.add(Activation('sigmoid'))
	rmsprop = RMSprop(lr=0.0008, rho=0.9, epsilon=1e-06)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
	print '=================='
	return model

def glove_init_LSTM(max_features, maxlen, args, d_output, tobetrain, embedding_matrix):
	embedding_dim = args.embed_dim
	d_lstm = args.lstm_dim
	d_dense = args.dense_dim
	print 'Building model'
	model = Sequential()
	model.add(Embedding(max_features, embedding_dim, weights = [embedding_matrix],input_length=maxlen, init='glorot_uniform', trainable=True)) # modified 1115 HSIEH
	model.add(LSTM(output_dim = d_lstm, return_sequences=False, input_shape=(maxlen, embedding_dim), init='glorot_uniform', inner_init='orthogonal', inner_activation='hard_sigmoid', trainable = tobetrain))
	model.add(Dense(d_dense, init='glorot_uniform', trainable = True))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(d_dense, init='glorot_uniform', trainable = True))
	model.add(Activation('relu'))
	model.add(Dense(d_output))
	model.add(Activation('softmax'))
	rmsprop = RMSprop(lr=0.0008, rho=0.9, epsilon=1e-06)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop,metrics=['accuracy'])
	print '=================='
	return model

def mypretrainLSTM(max_features, maxlen, args, d_output, tobetrain):
	embedding_dim = args.embed_dim
	d_lstm = args.lstm_dim
	d_dense = args.dense_dim
	print 'Building normal LSTM model'
	model = Sequential()
	model.add(Embedding(max_features, embedding_dim, input_length=maxlen, init='glorot_uniform'))
	model.add(LSTM(output_dim = d_lstm, return_sequences=False, input_shape=(maxlen, embedding_dim), init='glorot_uniform', inner_init='orthogonal', inner_activation='hard_sigmoid', trainable = tobetrain)) #modify Hsieh
	model.add(Dense(d_dense, init='glorot_uniform', trainable = True))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(d_dense, init='glorot_uniform', trainable = True))
	model.add(Activation('relu'))
	model.add(Dense(d_output))
	model.add(Activation('softmax'))
	rmsprop = RMSprop(lr=0.0008, rho=0.9, epsilon=1e-06)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop,metrics=['accuracy'])
	#model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
	print '=================='
	return model

def myattenLSTM(max_features, maxlen, args, dim_output):
	dim_latent = args.embed_dim
	dim_lstm = args.lstm_dim
	dim_dense = args.dense_dim
	print 'Building atten LSTM model'
	main_input = Input(shape=(maxlen,), dtype='int32',name='input')
	embed_out = Embedding(input_dim = max_features+2, output_dim = dim_latent, input_length = maxlen, init = 'glorot_uniform')(main_input) # shape (batch_size, maxlen, dim_latent)
	lstm_out = LSTM(output_dim = dim_lstm, return_sequences = False, input_shape = (maxlen,dim_latent), init ='glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')(embed_out) # b_s, dim_lstm 
	lstm_drop = Dropout(0.1)(lstm_out)
	lstm_to_embed = Dense(dim_latent, init = 'glorot_uniform', activation='relu')(lstm_drop) # b_s, dim_latent
	repeatVec = RepeatVector(maxlen)(lstm_to_embed) # b_s, maxlen, dim_latent
	dot_embed_out = merge([repeatVec, embed_out],  mode='mul') # b_s, maxlen, dim_latent
	per_dot = Permute((2,1))(dot_embed_out) # b_s, dim_latent, maxlen
	e_i = Lambda(sum_along_time,sum_along_time_output_shape)(per_dot) # b_s, maxlen
	if args.atten_mode =='sigmoid':
		#model.add_node(Activation(normalize_sigmoid), name='a', input='e') # shape (1,maxlen)
		a_i = Activation('sigmoid')(e_i) #  b_s,maxlen
	else:
		a_i = Activation('softmax')(e_i) # b_s, maxlen
	repeat_a = RepeatVector(dim_latent)(a_i) # b_s, dim_latent, maxlen
	per_a = Permute((2,1))(repeat_a) # b_s, maxlen, dim_latent
	mul_a_embed = merge([per_a, embed_out],mode='mul') # alpha_i * V_i
	atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(mul_a_embed) # b_s, dim_latent
	onedense = Dense(dim_dense, init='glorot_uniform', activation='relu')(atten_out) # b_s, dim_dense
	final_out = Dense(dim_output,activation='softmax',name='output')(onedense)
	model = Model(input=[main_input],output=[final_out])
	#rmsprop = RMSprop(lr=0.0008, rho =0.9, epsilon=1e-06)
	model.compile(optimizer='rmsprop', loss={'output':'categorical_crossentropy'},metrics=['accuracy'])
	#model.compile(optimizer='rmsprop', loss={'output':'categorical_crossentropy'})
	return model
	

def count_MAP_total(y, pred, oov, valid_index):
	# true doc is in 1000
	total_AP = np.zeros(len(valid_index))
	total_PR = np.zeros(len(valid_index))
	nowline = 0 
	for i,ind in enumerate(valid_index):
		if ind == False:
			total_AP[i] = 0
			total_PR[i] = 0
		else:
			total_AP[i], total_PR[i] = mymap(y[nowline,0:], pred[nowline, 0:], oov[i])
			nowline += 1
	return np.sum(total_AP)/len(valid_index), np.sum(total_PR)/len(valid_index)

def count_pr_total(y, pred, oov, valid_index):
	total_prec = np.zeros(len(valid_index))
	total_reca = np.zeros(len(valid_index))
	nowline = 0
	for i, ind in enumerate(valid_index):
		if ind == False:
			total_prec[i] = 0
			total_reca[i] = 0
		else:
			total_prec[i], total_reca[i] = multi_pr(y[nowline, 0:], pred[nowline, 0:], oov[i])
			nowline += 1
	return float(np.sum(total_prec))/len(valid_index), float(np.sum(total_reca))/len(valid_index)
	#return (total_prec), (total_reca)

def multi_pr(truth, pred, num_oov):
	true_doc_id = np.nonzero(truth)[0]
	pred[pred>=0.5]=1
	pred[pred<0.5]=0
	my_pred = np.nonzero(pred)[0]
	correct = 0
	one_prec = 0
	one_reca = 0
	for i in my_pred:
		if i in true_doc_id:
			correct += 1
	if not len(my_pred)==0:
		one_prec = float(correct)/(len(my_pred))
	one_reca = float(correct)/(len(true_doc_id)+num_oov)
	#another_accu = float(np.sum(np.equal(pred,truth)))/len(pred)
	return one_prec, one_reca

def mymap(truth, one_pred, num_oov):
	true_doc_id = np.nonzero(truth)[0]
	my_retrieve = one_pred.argsort()[::-1]
	num_to_retr = len(true_doc_id)
	correct = 0
	wrong = 0
	total_precision = []
	hit_times = 0
	for wtf,i in enumerate(my_retrieve):
		if i in true_doc_id:
		#	print('position '+str(wtf)+' hit index '+str(i))
			hit_times += 1
			correct += 1
			num_to_retr -= 1
			now_prec = float(correct)/(correct+wrong)
			total_precision.append(now_prec)
		#	print('Precision is '+str(now_prec))
		else:
			wrong += 1
		if (num_to_retr == 0) and (num_oov==0) :
			oop = float(sum(total_precision))/hit_times
			now_prec = float(correct)/(correct+wrong)
			return oop, now_prec
	while num_oov!=0:
		hit_times += 1
		correct += 1
		now_prec = float(correct)/(correct+wrong)
		total_precision.append(now_prec)
	#	print('OOV Precision is '+str(now_prec))
		oop = float(sum(total_precision))/hit_times
		num_oov -= 1
	return oop, now_prec
