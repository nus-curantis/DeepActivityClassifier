import tensorflow as tf

import os
import sys
sys.path.append("../../DeepActivityClassifier")

from deep_model.deep_conv_lstm_classifier_complex_cnn_52 import DeepConvLSTMClassifier

# model parameters
# tf.flags.DEFINE_integer('input_representations', 3, 'num of input representations')
# tf.flags.DEFINE_integer('input_representations', 6, 'num of input representations')  # Accelerometer and gyroscope
tf.flags.DEFINE_integer('input_representations', 52, 'num of input representations')

tf.flags.DEFINE_integer('num_classes', 13, 'num of classes in the output')  # mhealth dataset and pamap2
# tf.flags.DEFINE_integer('num_classes', 5, 'num of classes in the output')  # our dataset after ignoring some classes
# tf.flags.DEFINE_integer('num_classes', 19, 'num of classes in the output')  # our dataset
# tf.flags.DEFINE_integer('num_classes', 18, 'num of classes in the output')  # our dataset
# tf.flags.DEFINE_integer('num_classes', 8, 'num of classes in the output')  # chest accelerometer data
# tf.flags.DEFINE_integer('num_classes', 14, 'num of classes in the output')  # wharf data

# tf.flags.DEFINE_integer('series_max_len', 360, 'max len of an input time series')
tf.flags.DEFINE_integer('series_max_len', 180, 'max len of an input time series')

# tf.flags.DEFINE_integer('rnn_hidden_units', 32, 'hidden neurons of rnn cells')
# tf.flags.DEFINE_integer('rnn_hidden_units', 64, 'hidden neurons of rnn cells')
tf.flags.DEFINE_integer('rnn_hidden_units', 256, 'hidden neurons of rnn cells')

tf.flags.DEFINE_integer('filter_1_x', 6, 'conv layer 1 - filter x dim')
tf.flags.DEFINE_integer('filter_1_y', 6, 'conv layer 1 - filter y dim')
tf.flags.DEFINE_integer('filters_num_1', 30, 'num of filters in conv layer 1')
tf.flags.DEFINE_integer('stride_1_x', 3, 'conv layer 1 - filter movement in x dim')
tf.flags.DEFINE_integer('stride_1_y', 3, 'conv layer 1 - filter movement y dim')

tf.flags.DEFINE_integer('filter_2_x', 6, 'conv layer 2 - filter x dim')
tf.flags.DEFINE_integer('filter_2_y', 6, 'conv layer 2 - filter y dim')
tf.flags.DEFINE_integer('filters_num_2', 30, 'num of filters in conv layer 2')
tf.flags.DEFINE_integer('stride_2_x', 3, 'conv layer 2 - filter movement in x dim')
tf.flags.DEFINE_integer('stride_2_y', 3, 'conv layer 2 - filter movement y dim')

tf.flags.DEFINE_integer('filter_3_x', 3, 'conv layer 3 - filter x dim')
tf.flags.DEFINE_integer('filter_3_y', 3, 'conv layer 3 - filter y dim')
tf.flags.DEFINE_integer('filters_num_3', 20, 'num of filters in conv layer 3')
tf.flags.DEFINE_integer('stride_3_x', 3, 'conv layer 3 - filter movement in x dim')
tf.flags.DEFINE_integer('stride_3_y', 3, 'conv layer 3 - filter movement y dim')

tf.flags.DEFINE_integer('filter_4_x', 3, 'conv layer 4 - filter x dim')
tf.flags.DEFINE_integer('filter_4_y', 3, 'conv layer 4 - filter y dim')
tf.flags.DEFINE_integer('filters_num_4', 20, 'num of filters in conv layer 4')
tf.flags.DEFINE_integer('stride_4_x', 3, 'conv layer 4 - filter movement in x dim')
tf.flags.DEFINE_integer('stride_4_y', 3, 'conv layer 4 - filter movement y dim')

tf.flags.DEFINE_float('dropout_prob', .9, 'drop out keep probability')

# learning parameters
tf.flags.DEFINE_float('learning_rate', .001, 'learning rate')
tf.flags.DEFINE_string('activation_func', 'relu', 'activation function')
# tf.flags.DEFINE_integer('num_epochs', 150, 'number of training epochs')
tf.flags.DEFINE_integer('num_epochs', 300, 'number of training epochs')
# tf.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')

# logging parameters
tf.flags.DEFINE_string('log_folder', 'logs_classifier_new/52_4d', 'tensorboard logs folder')
tf.flags.DEFINE_string('model_path', './deep_model_weights', 'saved model folder')

config = tf.flags.FLAGS

# Setting the device to GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = DeepConvLSTMClassifier(config=config)
model.build_model()
model.train()
