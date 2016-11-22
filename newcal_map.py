import argparse
import cPickle as pickle
import sys 
import numpy as np

from sklearn.metrics import average_precision_score

## my own library
from my_utils import printParams, myloadData, mypretrainLSTM, myattenLSTM 
from my_utils import get_dict, vectorize_label, mymap, count_MAP_total

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
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-train_on",type=str,required=True, choices=['stack','interspeech'])
parser.add_argument("-weight",type=str, required=True, help="Required by cal_map")
args = parser.parse_args()

file_name_test = vectorpathin+source+'.test.body.vector'
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

if args.train_on == 'stack':
	d_output = 1000
else:
	d_output = 1500


print '=================='
print 'Cal Map on  :',args.train_on
print '=================='
printParams(args,max_features,maxlen,d_output)

# build Model
if args.atten_mode == 'no':
	source_model = mypretrainLSTM(max_features,maxlen,args,d_output,True)
else:
	source_model = myattenLSTM(max_features,maxlen,args,d_output)


X_test, Y_test = myloadData(file_name_test,file_name_test_tag,d_output,maxlen)
Y_test = Y_test.astype(np.float32)

print 'Showing MAP result of ',source
print 'Using Pretrain Weight :',args.weight

coco = pickle.load(open(args.weight, 'rb'))
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

	
