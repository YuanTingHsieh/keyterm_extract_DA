import argparse
import cPickle as pickle
import sys 
import numpy as np

from sklearn.metrics import average_precision_score

## my own library
from my_utils import printParams, myloadDataNaive, Naive 
from my_utils import get_dict, vectorize_label, mymap, count_MAP_total

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
#parser.add_argument("-ext_mode", type=str, default='freq', help="Choose extract mode: freq or tfidf", choices=['freq','tfidf'])
parser.add_argument("-tag_num", type=int, default=1000)
parser.add_argument("-embed_dim", type=int, default=300)
parser.add_argument("-dense_dim", type=int, default=1024)
parser.add_argument("-lstm_dim", type=int, default=128)
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-mode", type=str, required=True, help="Choose to train model or pred and cal map", choices=['train','cal_map'])
parser.add_argument("-exp_name", type=str, required=True, help="Name this experiment!!")
parser.add_argument("-train_on",type=str,required=True, choices=['stack','interspeech'])
#parser.add_argument("-max_features", type=int, default=15000)
args = parser.parse_args()

# paths
'''common paths'''
source = args.train_on
#source = 'stack'
#target = 'interspeech'
datapathin = './splitdata/'
vectorpathin = './splitvector/'
pathweight = './weight/'
weightname = source+'_'+args.exp_name+'.pretrain.weight'


'''train paths'''
file_name_train = vectorpathin+source+'.train.body.vector'
file_name_test = vectorpathin+source+'.test.body.vector'
file_name_train_tag = vectorpathin+source+'.train.tag.vector'
file_name_test_tag = vectorpathin+source+'.test.tag.vector'
file_dic_name = datapathin+'All.dic.body'

'''cal_map paths'''
file_tag_dic = datapathin+source+'.dic.tag'
file_test_tag = datapathin+source+'.test.tag'
file_train_tag = datapathin+source+'.train.tag'
resultname = './result/'+source+'_'+args.exp_name+'.result'


# parameters
lines = open(file_dic_name,'r').read().splitlines()
max_features = len(lines) + 1

maxlen = 0
for oneline in open(file_name_train).read().splitlines():
	if len(oneline.split())>maxlen:
		maxlen = len(oneline.split())

if args.train_on == 'stack':
	d_output = 1000
else:
	d_output = 1500


print '=================='
print 'Pretrain On Source Domain :',source
print '=================='
printParams(args,max_features,maxlen,d_output)


# build Model
source_model = Naive(max_features,args, d_output)

# training
if args.mode == 'train':
	# load data
	X_train, Y_train = myloadDataNaive(file_name_train,file_name_train_tag, d_output, max_features)
	X_train =  X_train.astype(np.float32)
	Y_train =  Y_train.astype(np.float32)
	X_test, Y_test = myloadDataNaive(file_name_test,file_name_test_tag,d_output, max_features)
	X_test =  X_test.astype(np.float32)
	Y_test = Y_test.astype(np.float32)
	print 'Start training'
	
	word_indices_val, indices_word_val = get_dict(file_tag_dic) 
	y_val, y_val_norm, valid_val_index, oov_val = vectorize_label(file_test_tag, d_output, word_indices_val)
	y_val = y_val[valid_val_index,0:]
	y_val_norm = y_val_norm[valid_val_index,0:]

	y, y_norm, valid_index, oov = vectorize_label(file_train_tag, d_output, word_indices_val)
	y = y[valid_index,0:]
	y_norm = y_norm[valid_index,0:]

	print "Training started..."
	
	for e in range(args.epochs):
		print "================================================Epoch %d================================================================" % (e+1)
		source_model.fit(
			X_train,
			Y_train,
			batch_size=args.batch_size,
			nb_epoch=1,
			verbose=1,
			validation_data=(X_test, Y_test)
		)
		#loss, accuracy = source_model.evaluate(X_train, Y_train,verbose=0,batch_size=args.batch_size)
		#print "loss is %f , accuracy is %f" %(loss,accuracy)
		#loss, accuracy = source_model.evaluate(X_test, Y_test,verbose=0,batch_size=args.batch_size)
		#print "val_loss is %f , val_accuracy is %f" %(loss,accuracy)
		pred_val = source_model.predict(X_test)
		pred_train = source_model.predict(X_train)
	
		map_oov, pr_oov = count_MAP_total(y, pred_train, oov, valid_index)
		print('MAP of train(oov keyword included) is ',map_oov)

		map_no_oov, pr_no_oov  = count_MAP_total(y, pred_train, np.zeros(len(oov)), valid_index)
		print('MAP of train(oov keyword not included) is ',map_no_oov)

		map_oov, pr_oov = count_MAP_total(y_val, pred_val, oov_val, valid_val_index)
		print('MAP of validaton(oov keyword included) is ',map_oov)

		map_no_oov, pr_no_oov  = count_MAP_total(y_val, pred_val, np.zeros(len(oov_val)), valid_val_index)
		print('MAP of validaton(oov keyword not included) is ',map_no_oov)

		print "======================================================================================================================="
		# write out weights
		if (e+1)%10 == 0:
			theweight = source_model.get_weights()
			fileout = open(pathweight + weightname + '_epo'+str(e+1),'wb')
			pickle.dump(theweight, fileout)
	
else:	
	X_test, Y_test = myloadData(file_name_test,file_name_test_tag,d_output,maxlen)
	Y_test = Y_test.astype(np.float32)
	weightname += ('_epo'+str(args.epochs))

	print 'Showing MAP result of ',source
	print 'Using Pretrain Weight :',weightname

	coco = pickle.load(open(pathweight+weightname,'rb'));
	source_model.set_weights(coco)
	pred_val = source_model.predict(X_test)
	

	word_indices_val, indices_word_val = get_dict(file_tag_dic) 
	y_val, y_val_norm, valid_val_index, oov_val = vectorize_label(file_test_tag, d_output, word_indices_val)
	y_val = y_val[valid_val_index,0:]
	y_val_norm = y_val_norm[valid_val_index,0:]

	

	map_oov, pr_oov = count_MAP_total(y_val, pred_val, oov_val, valid_val_index)
	print('MAP of validaton(oov keyword included) is ',map_oov)

	map_no_oov, pr_no_oov  = count_MAP_total(y_val, pred_val, np.zeros(len(oov_val)), valid_val_index)
	print('MAP of validaton(oov keyword not included) is ',map_no_oov)

	
