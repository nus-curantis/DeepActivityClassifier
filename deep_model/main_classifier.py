import tensorflow as tf
import os
from deep_model.deep_activity_classifier import DeepActivityClassifier

# model parameters
tf.flags.DEFINE_integer('input_representations', 3, 'num of input representations')
# tf.flags.DEFINE_integer('input_representations', 1, 'num of input representations')
tf.flags.DEFINE_integer('num_classes', 18, 'num of classes in the output')
tf.flags.DEFINE_integer('embedding_out_size', 5, 'embedding layer output features')
tf.flags.DEFINE_integer('series_max_len', 360, 'max len of an input time series')
# tf.flags.DEFINE_integer('rnn_hidden_units', 16, 'hidden neurons of rnn cells')
tf.flags.DEFINE_integer('rnn_hidden_units', 32, 'hidden neurons of rnn cells')

# learning parameters
tf.flags.DEFINE_float('learning_rate', 1, 'learning rate')
tf.flags.DEFINE_string('activation_func', 'relu', 'activation function')
# tf.flags.DEFINE_integer('num_epochs', 150, 'number of training epochs')
tf.flags.DEFINE_integer('num_epochs', 450, 'number of training epochs')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
# tf.flags.DEFINE_integer('batch_size', 32, 'batch size')
# tf.flags.DEFINE_integer('batch_size', 256, 'batch size')

# logging parameters
tf.flags.DEFINE_string('log_folder', 'logs_classifier', 'tensorboard logs folder')
tf.flags.DEFINE_string('model_path', './deep_model_weights', 'saved model folder')

config = tf.flags.FLAGS

# Setting the device to GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = DeepActivityClassifier(config=config)
model.build_model()
model.train()
