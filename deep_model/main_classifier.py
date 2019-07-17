import tensorflow as tf
import os
# from deep_model.deep_activity_classifier import DeepActivityClassifier
from deep_model.deep_conv_lstm_classifier import DeepConvLSTMClassifier

# model parameters
tf.flags.DEFINE_integer('input_representations', 3, 'num of input representations')
# tf.flags.DEFINE_integer('num_classes', 18, 'num of classes in the output')  # our dataset
# tf.flags.DEFINE_integer('num_classes', 8, 'num of classes in the output')  # chest accelerometer data
tf.flags.DEFINE_integer('num_classes', 14, 'num of classes in the output')  # wharf data
tf.flags.DEFINE_integer('embedding_out_size', 5, 'embedding layer output features')
# tf.flags.DEFINE_integer('series_max_len', 360, 'max len of an input time series')
# tf.flags.DEFINE_integer('series_max_len', 180, 'max len of an input time series')
tf.flags.DEFINE_integer('series_max_len', 90, 'max len of an input time series')
# tf.flags.DEFINE_integer('rnn_hidden_units', 32, 'hidden neurons of rnn cells')
tf.flags.DEFINE_integer('rnn_hidden_units', 64, 'hidden neurons of rnn cells')
tf.flags.DEFINE_integer('split_len', 6,
                        'indicates input split len. The split segments go throw a conv layer before entering rnn')
tf.flags.DEFINE_integer('filters_num', 5, 'num of filters in conv net')
# tf.flags.DEFINE_integer('filters_num', 10, 'num of filters in conv net')
tf.flags.DEFINE_float('dropout_prob', .75, 'drop out keep probability')


# learning parameters
tf.flags.DEFINE_float('learning_rate', 1, 'learning rate')
# tf.flags.DEFINE_float('learning_rate', .5, 'learning rate')
tf.flags.DEFINE_string('activation_func', 'relu', 'activation function')
tf.flags.DEFINE_integer('num_epochs', 150, 'number of training epochs')
# tf.flags.DEFINE_integer('num_epochs', 100, 'number of training epochs')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')

# logging parameters
tf.flags.DEFINE_string('log_folder', 'logs_classifier', 'tensorboard logs folder')
tf.flags.DEFINE_string('model_path', './deep_model_weights', 'saved model folder')

config = tf.flags.FLAGS

# Setting the device to GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# model = DeepActivityClassifier(config=config)
model = DeepConvLSTMClassifier(config=config)
model.build_model()
model.train()
