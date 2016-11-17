import argparse
import cPickle as pickle
import sys 
import numpy as np
import os.path

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
parser.add_argument("-embed_dim", type=int, default=300)
parser.add_argument("-dense_dim", type=int, default=1024)
parser.add_argument("-lstm_dim", type=int, default=128)
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-mode", type=str, required=True, help="Choose to train model or pred and cal map", choices=['train','cal_map'])
parser.add_argument("-exp_name", type=str, required=True, help="Name this experiment!!")
#parser.add_argument("-max_features", type=int, default=15000)
parser.add_argument("-weight", type=str)
args = parser.parse_args()

# paths
'''common paths'''
source = 'stack'
target = 'interspeech'
datapathin = './splitdata/'
vectorpathin = './splitvector/'
pathweight = './weight/'
weightname = source+'_'+args.exp_name+'.alter.weight'
weightname_target = target+'_'+args.exp_name+'.alter.weight'
source_tag_num = 1000
target_tag_num = 1500


'''train paths'''
file_name_train = vectorpathin+source+'.train.body.vector'
file_name_test = vectorpathin+source+'.test.body.vector'
file_name_train_tag = vectorpathin+source+'.train.tag.vector'
file_name_test_tag = vectorpathin+source+'.test.tag.vector'
file_dic_name = datapathin+'All.dic.body'
'''target train paths'''
file_name_train_target = vectorpathin+target+'.train.body.vector'
file_name_test_target = vectorpathin+target+'.test.body.vector'
file_name_train_tag_target = vectorpathin+target+'.train.tag.vector'
file_name_test_tag_target = vectorpathin+target+'.test.tag.vector'


'''cal_map paths'''
file_tag_dic = datapathin+source+'.dic.tag'
file_test_tag = datapathin+source+'.test.tag'
#resultname = './result/'+source+'_'+args.exp_name+'.result'
file_tag_dic_target = datapathin+target+'.dic.tag'
file_test_tag_target = datapathin+target+'.test.tag'


# parameters
lines = open(file_dic_name,'r').read().splitlines()
max_features = len(lines) + 1

maxlen = 0
for oneline in open(file_name_train):
	if len(oneline.split())>maxlen:
		maxlen = len(oneline.split())

d_output = source_tag_num


maxlen_target = 0
for oneline in open(file_name_train_target):
	if len(oneline.split())>maxlen_target:
		maxlen_target = len(oneline.split())

d_output_target = target_tag_num

print '========================='
print 'Doing Alternating Training'
print '========================='
print 'Source Info'
printParams(args,max_features,maxlen,d_output)
print '========================='
print 'Target Info'
printParams(args,max_features,maxlen_target,d_output_target)

source_model = mypretrainLSTM(max_features,maxlen,args,d_output,True)
target_model = mypretrainLSTM(max_features,maxlen_target,args,d_output_target,False)

if args.mode == 'train':
	#load data
	X_train, Y_train = myloadData(file_name_train,file_name_train_tag,d_output,maxlen)
	Y_train =  Y_train.astype(np.float32)
	X_test, Y_test = myloadData(file_name_test,file_name_test_tag,d_output,maxlen)
	Y_test = Y_test.astype(np.float32)

	X_train_target, Y_train_target = myloadData(file_name_train_target,file_name_train_tag_target,d_output_target,maxlen_target)
	Y_train_target =  Y_train_target.astype(np.float32)
	X_test_target, Y_test_target = myloadData(file_name_test_target,file_name_test_tag_target,d_output_target,maxlen_target)
	Y_test_target = Y_test_target.astype(np.float32)

	word_indices, indices_word = get_dict(file_tag_dic) 
	y, y_norm, valid_index, oov = vectorize_label(file_test_tag, d_output, word_indices)
	y = y[valid_index,0:]
	y_norm = y_norm[valid_index,0:]

	word_indices_val, indices_word_val = get_dict(file_tag_dic_target) 
	y_val, y_val_norm, valid_val_index, oov_val = vectorize_label(file_test_tag_target, d_output_target, word_indices_val)
	y_val = y_val[valid_val_index,0:]
	y_val_norm = y_val_norm[valid_val_index,0:]

	print 'Start Training'
	for e in range(args.epochs):
		print "================================================Epoch %d================================================================" % (e+1)
		if (os.path.isfile(pathweight + weightname_target + '_epo'+str(e-1))):
			filein = open(pathweight + weightname_target + '_epo'+str(e-1),'rb')
			coco = pickle.load(filein)
			coco = coco[:-5]
			oldweight = source_model.get_weights()
			coco.append(oldweight[-5])
			coco.append(oldweight[-4])
			coco.append(oldweight[-3])
			coco.append(oldweight[-2])
			coco.append(oldweight[-1])
			source_model.set_weights(coco)
			filein.close()

		source_model.fit(
			X_train,
			Y_train,
			batch_size=args.batch_size,
			nb_epoch=1,
			#verbose=0
			validation_data=(X_test, Y_test)
		)
		
		pred = source_model.predict(X_test)
	
		theweight = source_model.get_weights()
		fileout = open(pathweight + weightname + '_epo'+str(e),'wb')
		pickle.dump(theweight, fileout)
		fileout.close()

		oldweight = target_model.get_weights()
		theweight = theweight[:-5]
		theweight.append(oldweight[-5])
		theweight.append(oldweight[-4])
		theweight.append(oldweight[-3])
		theweight.append(oldweight[-2])
		theweight.append(oldweight[-1])
		target_model.set_weights(theweight)
		target_model.fit(
			X_train_target,
			Y_train_target,
			batch_size=args.batch_size,
			nb_epoch=1,
			#verbose=0
			validation_data=(X_test_target, Y_test_target)
		)

		theweight = target_model.get_weights()
		fileout = open(pathweight + weightname_target + '_epo'+str(e),'wb')
		pickle.dump(theweight, fileout)
		fileout.close()
		
		pred_val = target_model.predict(X_test_target)

		if e%2 == 0:
			map_oov, pr_oov = count_MAP_total(y, pred, oov, valid_index)
			print('MAP of train(oov keyword included) is ',map_oov)
			#print('P@R of train(oov keyword included) is ',pr_oov)

			map_no_oov, pr_no_oov  = count_MAP_total(y, pred, np.zeros(len(oov)), valid_index)
			print('MAP of train(oov keyword not included) is ',map_no_oov)
			#print('P@R of train(oov keyword not included) is ',pr_no_oov)

			map_oov_val, pr_oov_val = count_MAP_total(y_val, pred_val, oov_val, valid_val_index)
			print('MAP of target(oov keyword included) is ',map_oov_val)
			#print('P@R of target(oov keyword included) is ',pr_oov_val)

			map_no_oov_val, pr_no_oov_val  = count_MAP_total(y_val, pred_val, np.zeros(len(oov_val)), valid_val_index)
			print('MAP of target(oov keyword not included) is ',map_no_oov_val)
			#print('P@R of target(oov keyword not included) is ',pr_no_oov_val)
