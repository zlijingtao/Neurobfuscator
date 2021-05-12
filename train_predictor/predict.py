#!/usr/bin/env python

import time
import argparse
import os
from multiprocessing import cpu_count
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import logging
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import csv
import pandas as pd
import pickle
from util import infolog, plot, ValueWindow

from tensorflow.python import pywrap_tensorflow

from getSample import SampleFinder, sparse_tuple_from, all_select_seg

indexDict = {'conv':0, 'fc':1, 'pooling':2, 'bn':3, 'depthConv':4, 'relu':5, 'pointConv':6, 'add':7, 'softmax': 8}
layer_int_to_name_map = {0:'conv', 1:'fc', 2:'pooling', 3:'bn', 4:'depthConv',5:'relu', 6:'pointConv', 7:'add',8:'softmax'}

def convert_inputs_to_ctc_format(sampleName):
    # print(target_layers)

    #inputs are reading a sequence of the profiling log
    #it is two dimensiones:  k:(latency, r, w, r/w , i/o)
    #(k1,k2,...,kn)

    inputs = SampleFinder(sampleName)
    # Transform in 3D array
    if inputs == []:
        return [],[],[]
    train_inputs = np.array(inputs)
    # print(train_inputs)

    # train_inputs = (train_inputs - feature_mean) / feature_std
    train_seq_len = [train_inputs.shape[0]]

    return train_inputs, train_seq_len
    #, original

def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
    return varlist

slim = tf.contrib.slim

log = infolog.log

# Some configs. (latency, r, w, r/w, i/o)




def add_stats(cost, ler):
  with tf.variable_scope('stats') as scope:
    tf.summary.scalar('val_cost', cost)  # FIXME: now only feed with val data
    tf.summary.scalar('val_ler', ler)
    return tf.summary.merge_all()



def next_infer_batch(sample_file):
    row = ''
    # from getSample import convert_inputs_to_ctc_format
    infer_inputs, infer_seq_len = convert_inputs_to_ctc_format(sample_file)
    z = np.zeros((1, infer_inputs.shape[0], 1), dtype=infer_inputs.dtype)
    infer_inputs = np.reshape(infer_inputs,(1,infer_inputs.shape[0],5))
    infer_inputs = np.concatenate((infer_inputs, z), axis = 2)
    return infer_inputs, infer_seq_len


def load_trace_raw(csv_file, num_features = 5, normalize = "sb"):
    if num_features == 5:
        _, train_inputs = trace_csv_numpy(csv_file)
    elif num_features == 11:
        train_inputs, _ = trace_csv_numpy(csv_file)
    elif num_features == 1:
        _, train_inputs = trace_csv_numpy(csv_file)
        train_inputs = train_inputs[:,:,:1]
    # z = np.zeros((1, train_inputs.shape[1], 1), dtype=train_inputs.dtype)
    # train_inputs = np.concatenate((train_inputs, z), axis = 2)
    if normalize == "sb":
        mean = np.mean(train_inputs)
        std = np.std(train_inputs)
        train_inputs = (train_inputs - mean)/std
    elif normalize == "smart":
        norm = np.linalg.norm(train_inputs, axis = 1)
        mean = np.mean(train_inputs, axis = 1)
        # train_inputs = np.divide(train_inputs,norm)
        train_inputs = np.nan_to_num((train_inputs- mean)/norm)
    train_seq_len = [train_inputs.shape[1]]
    return train_inputs, train_seq_len

def load_label_raw(npy_file):
    train_labels = np.load(npy_file)
    return train_labels

def trace_csv_numpy(trace_file = "None"):
    full_trace_array = None
    reduced_trace_array = None
    df = pd.read_csv(trace_file, skiprows=2)
    trace_df = df[['ID', 'Metric Name', 'Metric Value']]
    reduced_trace_array = np.zeros((1, trace_df['ID'].nunique(), 5))
    full_trace_array = np.zeros((1, trace_df['ID'].nunique(), trace_df['Metric Name'].nunique()))
    old_row_id = -1
    count = 0
    for index, row in trace_df.iterrows():
        if row['ID'] == old_row_id:
            count += 1
        else:
            old_row_id += 1
            count = 0
        full_trace_array[0, old_row_id, count] = row['Metric Value']
        if row['Metric Name'] == 'Cycles':
            reduced_trace_array[0, old_row_id, 0] = row['Metric Value']
        elif row['Metric Name'] == 'Mem Read':
            reduced_trace_array[0, old_row_id, 1] = row['Metric Value']
        elif row['Metric Name'] == 'Mem Write':
            reduced_trace_array[0, old_row_id, 2] = row['Metric Value']
    reduced_trace_array[0, 0, 4] = 1.0
    for i in range(trace_df['ID'].nunique()):
        if reduced_trace_array[0, i, 2] != 0:
            reduced_trace_array[0, i, 3] = reduced_trace_array[0, i, 1]/reduced_trace_array[0, i, 2]
        if i > 0:
            if reduced_trace_array[0, i-1, 2] != 0:
                reduced_trace_array[0, i, 4] = reduced_trace_array[0, i, 1]/reduced_trace_array[0, i-1, 2]
            else:
                reduced_trace_array[0, i, 4] = 1.0
    full_trace_array = np.nan_to_num(full_trace_array)
    reduced_trace_array = np.nan_to_num(reduced_trace_array)
    return full_trace_array, reduced_trace_array

def save0_1_insparse_todense(sp_tensor):
    #Reward Function Related
    ds_tensor = tf.sparse.to_dense(sp_tensor)
    conv_index = tf.where(tf.equal(ds_tensor, 0))
    fc_index = tf.where(tf.equal(ds_tensor, 1))
    depthconv_index = tf.where(tf.equal(ds_tensor, 4))
    pointconv_index = tf.where(tf.equal(ds_tensor, 6))
    softmax_index = tf.where(tf.equal(ds_tensor, 8))
    complex_index = tf.concat(axis=0,values=[conv_index, fc_index, depthconv_index, pointconv_index, softmax_index])
    fi_tensor = tf.reshape(tf.gather_nd(ds_tensor, complex_index), [-1, 1])
    return fi_tensor

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

def run_ctc(num_features, normalize, log_dir, args):
    # OP classes + blank + others (10)
    num_classes = 10

    # Hyper-parameters
    num_epochs = 100   #10000
    num_hidden = 128
    num_layers = 1

    batch_size = 1
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    log('Checkpoint path: %s' % checkpoint_path)

    log('Using model: %s' % args.model_path)

    # Build the model
    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features]) #batch size = 1
       
        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])

        # Defining the cell
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        
        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                            state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)


        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits_f = tf.transpose(logits, (1, 0, 2))

        optimizer = tf.train.AdamOptimizer(100)
        
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_f, seq_len, merge_repeated=False)

        # Inaccuracy: label error rate
        decoded_tensor = tf.cast(decoded[0], tf.int32)

        prediction = save0_1_insparse_todense(decoded_tensor)

        flattened_predict = tf.reshape(prediction, [-1])

    # Bookkeeping:
    time_window = ValueWindow(100)
    train_cost_window = ValueWindow(100)
    train_ler_window = ValueWindow(100)
    val_cost_window = ValueWindow(100)
    val_ler_window = ValueWindow(100)

    # Run!
    with tf.Session(graph=graph) as sess:
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        if args.restore_step:
            # Restore from a checkpoint if the user requested it.
            restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
            variables = slim.get_variables_to_restore()
            variables_to_restore = [v for v in variables if v.name.split('/')[0]!='boosting']
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, restore_path)
            log('Resuming from checkpoint: %s' % (restore_path,))
        else:
            log('Starting new training run')


        for index_val in range(0,1):
            index_val = index_val + 1
            sample_file = args.sample_file
            if "csv" in sample_file:
                val_inputs, val_seq_len  = load_trace_raw(sample_file, num_features, normalize)
            else:
                val_inputs, val_seq_len = next_infer_batch(sample_file)
                
            print(val_inputs)

            val_feed = {inputs: val_inputs,
                        seq_len: val_seq_len}
            # Decoding
            d, lo = sess.run([decoded, logits_f], feed_dict=val_feed)
            # print(lo)
            # Replacing blank label to none
            str_decoded = ''
            for x in np.asarray(d[0][1]):
                if x in layer_int_to_name_map:
                    str_decoded = str_decoded + layer_int_to_name_map[x] + ' '
                else:
                    print("x=%d MAJOR ERROR? OUT OF PREDICTION SCOPE" % x)
            print('Decoded val:')
            print(str_decoded)

def run_model(normalize, num_features, num_classes, num_hidden, num_layers, log_dir, restore_step, label_file, sample_file):
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    #log(hparams_debug_string())
    
    # Set up DataFeeder:
    #dataset = DataFeeder(input_path)
    # Build the model
    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features]) #batch size = 1

        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])

        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
        # cell = tf.nn.rnn_cell.GRUCell(num_units = num_hidden)
        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                            state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        
        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits_f = tf.transpose(logits, (1, 0, 2))

        loss = - tf.nn.ctc_loss(targets, logits_f, seq_len, ctc_merge_repeated = False)
        
        # cost = tf.reduce_mean(loss) + 10*loss2
        cost = tf.reduce_mean(loss)
            
        optimizer = tf.train.AdamOptimizer(100)
        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_f, seq_len, merge_repeated=False)
        # decoded, log_prob = tf.nn.ctc_beam_search_decoder_v2(logits_f, seq_len, beam_width=1)
        # Inaccuracy: label error rate

        # decoded_tensor = tf.cast(decoded[0], tf.int32)

        # prediction = save0_1_insparse_todense(decoded_tensor)
        # flattened_predict = tf.reshape(prediction, [-1])
        # flattened_predict = targets
        # real_target = save0_1_insparse_todense(targets)
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))
        # ler = tf.reduce_mean(tf.edit_distance(tf.contrib.layers.dense_to_sparse(prediction), tf.contrib.layers.dense_to_sparse(real_target)))
        
        
        # ler = tf.reduce_mean(tf.edit_distance(tf.contrib.layers.dense_to_sparse(prediction), tf.contrib.layers.dense_to_sparse(real_target)))
        
        # Define reward here.
        # reward = 5.0 * tf.reduce_sum(tf.cast(tf.abs(tf.add(prediction[:tf.math.minimum(tf.size(prediction), tf.size(real_target)), :], - real_target[:tf.math.minimum(tf.size(prediction), tf.size(real_target)), :])), tf. float32))
        # reward += 10.0 * tf.math.abs(tf.cast((tf.size(prediction) - tf.size(real_target)), tf. float32))
        
  
    # Run!
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        restore_path = '%s-%d' % (checkpoint_path, restore_step)
        variables = slim.get_variables_to_restore()
        # variables_to_restore = [v for v in variables if v.name.split('/')[0]!='boosting'] 
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(variables)
        saver.restore(sess, restore_path)
        val_inputs, val_seq_len = load_trace_raw(sample_file, num_features, normalize)
        # print(val_inputs.shape)
        # print(val_seq_len.shape)
        val_targets = load_label_raw(label_file)
        # print(val_targets)
        # val_targets = [0,0,2,0,0,0,0,0,0,0,1,1,1,1,8]
        # np.save(label_file, val_targets)
        val_targets = sparse_tuple_from([val_targets])
        
        adv_feed = {inputs: val_inputs,
                targets: val_targets,
                seq_len: val_seq_len}
        
        atk_reward, nimg = sess.run([ler, decoded[0]], feed_dict=adv_feed)
        atk_predict = ''
        for x in np.asarray(nimg[1]):
            if x in layer_int_to_name_map:
                atk_predict = atk_predict + layer_int_to_name_map[x] + ' '
            else:
                print("x=%d MAJOR ERROR? OUT OF PREDICTION SCOPE" % x)
        # print(atk_predict)
    return atk_reward, atk_predict

class predictor(object):
    def __init__(self, log_dir, restore_step, label_file, sample_file, predict_type = "reduced", normalize = "smart", num_hidden = 128, num_layers = 1):
        np.random.seed(1234)
        tf.random.set_random_seed(1234)
        self.normalize = normalize
        self.num_features = 5
        if predict_type == "reduced":
            self.num_features = 5
        elif predict_type == "full":
            self.num_features = 11
        elif predict_type == "time_only":
            self.num_features = 1
        self.num_classes = 10
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.log_dir = log_dir
        self.restore_step = restore_step
        self.label_file = label_file
        self.sample_file = sample_file
    def get_reward(self):
        #state is the sample_file
        try:
            reward, predict = run_model(self.normalize, self.num_features, self.num_classes, self.num_hidden, self.num_layers,
                            self.log_dir, self.restore_step, self.label_file, self.sample_file)
        except:       
            reward = 0
            predict = "None"
        return reward, predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='first_ctc')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--predict_type', type=str, default="reduced", help='Pick dataset you want to predict on', choices=("reduced", "full", "time_only"))
    parser.add_argument('--normalize', type=str, default="sb", help='this model using which normalization', choices=("sb", "smart"))
    parser.add_argument('--summary_interval', type=int, default=1, help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Steps between writing checkpoints.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--sample_file', type=str)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    model_path = args.model_path

    log_dir = model_path
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'inference.log'), model_path)
    if args.predict_type == "reduced":
            num_features = 5
    elif args.predict_type == "time_only":
        num_features = 1
    elif args.predict_type == "full":
        num_features = 11
    run_ctc(num_features, args.normalize, log_dir, args)

if __name__ == '__main__':
    main()
