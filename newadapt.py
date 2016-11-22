import argparse
import cPickle as pickle
import sys 
import numpy as np

from sklearn.metrics import average_precision_score

## my own library
from my_utils import printParams, myloadData, mypretrainLSTM, myattenLSTM 
from my_utils import get_dict, vectorize_label, mymap, count_MAP_total

# tf mem control
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument("-atten_mode", type=str, required=True, help="Choose attention type: sigmoid or softmax or no", choices=['sigmoid','softmax','no'])
parser.add_argument("-embed_dim", type=int, default=300)
parser.add_argument("-dense_dim", type=int, default=1024)
parser.add_argument("-lstm_dim", type=int, default=128)
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-exp_name", type=str, required=True, help="Name this experiment!!")
parser.add_argument("-weight",type=str, required=True, help="Pretrain weight!")
parser.add_argument("-show_map",type=bool, help="Show map or not", default=False)
args = parser.parse_args()

# paths
'''common paths'''
source = 'interspeech'
datapathin = './splitdata/'
vectorpathin = './splitvector/'
pathweight = './adapt_weight/'
weightname = source+'_'+args.exp_name+'.weight'

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

# parameters
lines = open(file_dic_name,'r').read().splitlines()
max_features = len(lines) + 1

maxlen = 0
for oneline in open(file_name_train).read().splitlines():
	if len(oneline.split())>maxlen:
		maxlen = len(oneline.split())

if source == 'stack':
	d_output = 1000
else:
	d_output = 1500

print '=================='
print 'Adapt On Target Domain :',source
print '=================='
printParams(args,max_features,maxlen,d_output)

# build Model
if args.atten_mode == 'no':
	source_model = mypretrainLSTM(max_features,maxlen,args,d_output,True)
else:
	source_model = myattenLSTM(max_features,maxlen,args,d_output)

print 'Using Pretrain Weight :', args.weight
filein = open(args.weight, 'rb')
preweight = pickle.load(filein);
preweight = preweight[:-2]
oldweight = source_model.get_weights()
preweight.append(oldweight[-2])
preweight.append(oldweight[-1])
source_model.set_weights(preweight)
filein.close()

# load data
X_train, Y_train = myloadData(file_name_train,file_name_train_tag,d_output,maxlen)
Y_train =  Y_train.astype(np.float32)
X_test, Y_test = myloadData(file_name_test,file_name_test_tag,d_output,maxlen)
Y_test = Y_test.astype(np.float32)
	
word_indices_val, indices_word_val = get_dict(file_tag_dic) 
y_val, y_val_norm, valid_val_index, oov_val = vectorize_label(file_test_tag, d_output, word_indices_val)
y_val = y_val[valid_val_index,0:]
y_val_norm = y_val_norm[valid_val_index,0:]

y, y_norm, valid_index, oov = vectorize_label(file_train_tag, d_output, word_indices_val)
y = y[valid_index,0:]
y_norm = y_norm[valid_index,0:]

print 'Start training'
for e in range(args.epochs):
	print "================================================Epoch %d================================================================" % (e+1)
	source_model.fit(
		X_train,
		Y_train,
		batch_size=args.batch_size,
		nb_epoch=1,
		validation_data=(X_test, Y_test)
	)
		

	if args.show_map == True:
		pred_val = source_model.predict(X_test)
		pred_train = source_model.predict(X_train)
		map_oov, pr_oov = count_MAP_total(y_val, pred_val, oov_val, valid_val_index)
		map_no_oov, pr_no_oov  = count_MAP_total(y_val, pred_val, np.zeros(len(oov_val)), valid_val_index)
		print('MAP of validaton(oov keyword included) is ',map_oov)
		print('MAP of validaton(oov keyword not included) is ',map_no_oov)

	
	# write out weights
	theweight = source_model.get_weights()
	fileout = open(pathweight + weightname + '_epo'+str(e+1),'wb')
	pickle.dump(theweight, fileout)

pred_val = source_model.predict(X_test)
pred_train = source_model.predict(X_train)
map_oov, pr_oov = count_MAP_total(y, pred_train, oov, valid_index)
map_no_oov, pr_no_oov  = count_MAP_total(y, pred_train, np.zeros(len(oov)), valid_index)
print('MAP of train(oov keyword included) is ',map_oov)
print('MAP of train(oov keyword not included) is ',map_no_oov)
map_oov, pr_oov = count_MAP_total(y_val, pred_val, oov_val, valid_val_index)
map_no_oov, pr_no_oov  = count_MAP_total(y_val, pred_val, np.zeros(len(oov_val)), valid_val_index)
print('MAP of validaton(oov keyword included) is ',map_oov)
print('MAP of validaton(oov keyword not included) is ',map_no_oov)
