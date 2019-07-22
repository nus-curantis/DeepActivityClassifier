import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm

import numpy as np

# from pycm import ConfusionMatrix

from preprocessing.data_to_rnn_input_transformer import data_to_rnn_input_train_test, normalized_rnn_input_train_test
from preprocessing.wharf_reader import normalized_wharf_rnn_input_train_test


class DeepConvLSTMClassifier:
    def __init__(self, config):

        # model parameters
        self.embedding_out_size = config.embedding_out_size
        self.series_max_len = config.series_max_len
        self.rnn_hidden_units = config.rnn_hidden_units
        self.input_representations = config.input_representations
        self.num_classes = config.num_classes
        self.split_len = config.split_len
        self.filters_num = config.filters_num
        self.dropout_prob = config.dropout_prob

        # learning parameters
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        if config.activation_func == 'relu':
            self.activation_function = tf.nn.relu
        else:
            self.activation_function = tf.nn.tanh

        # logging parameters
        self.log_folder = config.log_folder
        self.model_path = config.model_path

        # inputs:
        self.input = tf.placeholder(tf.float32,
                                    shape=[None, self.series_max_len, self.input_representations])

        self.activity_label = tf.placeholder(tf.float32,
                                             shape=[None, self.num_classes])

        # survival model parts:
        self.embedding_layer = None
        self.embedded_input_x = None
        self.embedded_input_y = None
        self.embedded_input_z = None
        self.embedding = None
        self.conv_w_x = None
        self.conv_b_x = None
        self.conv_w_y = None
        self.conv_b_y = None
        self.conv_w_z = None
        self.conv_b_z = None
        self.rnn_cell = None
        self.rnn_output_x = None
        self.rnn_output_y = None
        self.rnn_output_z = None
        self.time_distributed_w_x = None
        self.time_distributed_b_x = None
        self.time_distributed_output_x = None
        self.time_distributed_w_y = None
        self.time_distributed_b_y = None
        self.time_distributed_output_y = None
        self.time_distributed_w_z = None
        self.time_distributed_b_z = None
        self.time_distributed_output_z = None
        self.avg_pooling_x = None
        self.max_pooling_x = None
        self.last_pooling_x = None
        self.avg_pooling_y = None
        self.max_pooling_y = None
        self.last_pooling_y = None
        self.avg_pooling_z = None
        self.max_pooling_z = None
        self.last_pooling_z = None
        self.concatenated_poolings = None
        self.dense_weights = None
        self.dense_biases = None
        self.hidden_layer_1 = None
        self.hidden_layer_2 = None
        self.prediction_logits = None
        self.prediction = None
        self.accuracy = None
        self.cost = None
        self.optimizer = None

        # log and save variables:
        self.loss_summary = None
        self.accuracy_summary = None
        self.train_summary = None
        self.validation_loss_summary = None
        self.validation_accuracy_summary = None
        self.validation_summary = None
        self.file_writer = None
        self.saver = None

        # loaded train, test data
        self.train_inputs = None
        self.test_inputs = None
        self.train_activity_labels = None
        self.test_activity_labels = None

    def set_data(self,
                 train_inputs,
                 test_inputs,
                 train_activity_labels,
                 test_activity_labels):
        self.train_inputs = train_inputs
        self.test_inputs = test_inputs
        self.train_activity_labels = train_activity_labels
        self.test_activity_labels = test_activity_labels

    def is_data_loaded(self):
        for data in [
            self.train_inputs,
            self.test_inputs,
            self.train_activity_labels,
            self.test_activity_labels,
        ]:
            if data is None:
                return False

        return True

    def load_data(self):
        self.train_inputs, self.test_inputs, self.train_activity_labels, self.test_activity_labels = \
            data_to_rnn_input_train_test(split_series_max_len=self.series_max_len)  # our dataset
            # normalized_rnn_input_train_test(data_path='../dataset/Chest_Accelerometer/data/',
            #                                 split_series_max_len=self.series_max_len)  # chest dataset
            # normalized_wharf_rnn_input_train_test(split_series_max_len=self.series_max_len)  # wahrf
            # data_to_rnn_input_train_test(data_path='../dataset/Chest_Accelerometer/data/')  # chest without normalizing

        print(len(self.train_inputs))
        print(len(self.train_activity_labels))
        print(len(self.test_inputs))
        print(len(self.test_activity_labels))

    def build_model(self):
        with tf.name_scope('cnn'):
            input_x = tf.expand_dims(self.input[:, :, 0], -1)
            input_y = tf.expand_dims(self.input[:, :, 1], -1)
            input_z = tf.expand_dims(self.input[:, :, 2], -1)

            self.conv_w_x = tf.Variable(tf.truncated_normal([self.split_len, 1, 1, self.filters_num]))
            self.conv_b_x = tf.Variable(tf.zeros([self.filters_num]))

            self.conv_w_y = tf.Variable(tf.truncated_normal([self.split_len, 1, 1, self.filters_num]))
            self.conv_b_y = tf.Variable(tf.zeros([self.filters_num]))

            self.conv_w_z = tf.Variable(tf.truncated_normal([self.split_len, 1, 1, self.filters_num]))
            self.conv_b_z = tf.Variable(tf.zeros([self.filters_num]))

            self.embedded_input_x = tf.nn.conv2d(input_x,
                                                 filter=self.conv_w_x,
                                                 strides=[1, self.split_len, 1, 1],
                                                 padding='VALID') + self.conv_b_x

            self.embedded_input_x = tf.reshape(self.embedded_input_x,
                                               shape=[-1,
                                                      self.embedded_input_x.shape[1] * self.embedded_input_x.shape[2],
                                                      self.filters_num])

            self.embedded_input_y = tf.nn.conv2d(input_y,
                                                 filter=self.conv_w_y,
                                                 strides=[1, self.split_len, 1, 1],
                                                 padding='VALID') + self.conv_b_y

            self.embedded_input_y = tf.reshape(self.embedded_input_y,
                                               shape=[-1,
                                                      self.embedded_input_y.shape[1] * self.embedded_input_y.shape[2],
                                                      self.filters_num])

            self.embedded_input_z = tf.nn.conv2d(input_z,
                                                 filter=self.conv_w_z,
                                                 strides=[1, self.split_len, 1, 1],
                                                 padding='VALID') + self.conv_b_z

            self.embedded_input_z = tf.reshape(self.embedded_input_z,
                                               shape=[-1,
                                                      self.embedded_input_z.shape[1] * self.embedded_input_z.shape[2],
                                                      self.filters_num])

        with tf.name_scope('rnn'):
            self.rnn_cell = rnn.LSTMCell(num_units=self.rnn_hidden_units)

            self.rnn_output_x, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell, inputs=self.embedded_input_x,
                dtype=tf.float32, sequence_length=self.__length(self.embedded_input_x))

            self.rnn_output_y, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell, inputs=self.embedded_input_y,
                dtype=tf.float32, sequence_length=self.__length(self.embedded_input_y))

            self.rnn_output_z, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell, inputs=self.embedded_input_z,
                dtype=tf.float32, sequence_length=self.__length(self.embedded_input_z))

        with tf.name_scope('time_distributed_layer'):
            self.time_distributed_w_x = tf.Variable(tf.truncated_normal([self.rnn_hidden_units, self.rnn_hidden_units]))
            self.time_distributed_b_x = tf.Variable(tf.zeros([self.rnn_hidden_units]))
            self.time_distributed_w_y = tf.Variable(tf.truncated_normal([self.rnn_hidden_units, self.rnn_hidden_units]))
            self.time_distributed_b_y = tf.Variable(tf.zeros([self.rnn_hidden_units]))
            self.time_distributed_w_z = tf.Variable(tf.truncated_normal([self.rnn_hidden_units, self.rnn_hidden_units]))
            self.time_distributed_b_z = tf.Variable(tf.zeros([self.rnn_hidden_units]))

            rnn_output_x_reshaped = tf.reshape(self.rnn_output_x, shape=[-1, self.rnn_hidden_units])
            time_distributed_output_x = tf.matmul(rnn_output_x_reshaped,
                                                  self.time_distributed_w_x) + self.time_distributed_b_x
            self.time_distributed_output_x = tf.reshape(time_distributed_output_x, shape=tf.shape(self.rnn_output_x))
            self.time_distributed_output_x = tf.nn.dropout(self.time_distributed_output_x, self.dropout_prob)

            rnn_output_y_reshaped = tf.reshape(self.rnn_output_y, shape=[-1, self.rnn_hidden_units])
            time_distributed_output_y = tf.matmul(rnn_output_y_reshaped,
                                                  self.time_distributed_w_y) + self.time_distributed_b_y
            self.time_distributed_output_y = tf.reshape(time_distributed_output_y, shape=tf.shape(self.rnn_output_y))
            self.time_distributed_output_y = tf.nn.dropout(self.time_distributed_output_y, self.dropout_prob)

            rnn_output_z_reshaped = tf.reshape(self.rnn_output_z, shape=[-1, self.rnn_hidden_units])
            time_distributed_output_z = tf.matmul(rnn_output_z_reshaped,
                                                  self.time_distributed_w_z) + self.time_distributed_b_z
            self.time_distributed_output_z = tf.reshape(time_distributed_output_z, shape=tf.shape(self.rnn_output_z))
            self.time_distributed_output_z = tf.nn.dropout(self.time_distributed_output_z, self.dropout_prob)

        with tf.name_scope('pooling'):
            self.avg_pooling_x = tf.reduce_mean(self.time_distributed_output_x, axis=1)
            self.max_pooling_x = tf.reduce_max(self.time_distributed_output_x, axis=1)
            self.last_pooling_x = self.__last_relevant(self.time_distributed_output_x,
                                                       self.__length(self.embedded_input_x))

            self.avg_pooling_y = tf.reduce_mean(self.time_distributed_output_y, axis=1)
            self.max_pooling_y = tf.reduce_max(self.time_distributed_output_y, axis=1)
            self.last_pooling_y = self.__last_relevant(self.time_distributed_output_y,
                                                       self.__length(self.embedded_input_y))

            self.avg_pooling_z = tf.reduce_mean(self.time_distributed_output_z, axis=1)
            self.max_pooling_z = tf.reduce_max(self.time_distributed_output_z, axis=1)
            self.last_pooling_z = self.__last_relevant(self.time_distributed_output_z,
                                                       self.__length(self.embedded_input_z))

            self.concatenated_poolings = tf.concat(
                [self.avg_pooling_x, self.max_pooling_x, self.last_pooling_x,
                 self.avg_pooling_y, self.max_pooling_y, self.last_pooling_y,
                 self.avg_pooling_z, self.max_pooling_z, self.last_pooling_z], axis=1
            )

        with tf.name_scope('predictor'):
            self.dense_weights = {
                'first': tf.Variable(tf.truncated_normal(
                    [9 * self.rnn_hidden_units, 6 * self.rnn_hidden_units])),
                'second': tf.Variable(tf.truncated_normal(
                    [6 * self.rnn_hidden_units, 3 * self.rnn_hidden_units])),
                'third': tf.Variable(tf.truncated_normal([3 * self.rnn_hidden_units, self.num_classes]))
            }

            self.dense_biases = {
                'first': tf.Variable(tf.zeros([6 * self.rnn_hidden_units])),
                'second': tf.Variable(tf.zeros([3 * self.rnn_hidden_units])),
                'third': tf.Variable(tf.zeros([self.num_classes])),
            }

            self.hidden_layer_1 = batch_norm(tf.matmul(
                self.concatenated_poolings, self.dense_weights['first']) + self.dense_biases['first'])

            self.hidden_layer_1 = self.activation_function(self.hidden_layer_1)

            self.hidden_layer_1 = tf.nn.dropout(self.hidden_layer_1, self.dropout_prob)

            self.hidden_layer_2 = batch_norm(tf.matmul(
                self.hidden_layer_1, self.dense_weights['second']) + self.dense_biases['second'])

            self.hidden_layer_2 = self.activation_function(self.hidden_layer_2)

            self.prediction_logits = tf.matmul(
                self.hidden_layer_2, self.dense_weights['third']) + self.dense_biases['third']

            self.prediction = tf.nn.softmax(self.prediction_logits)

            correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.activity_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope('prediction_optimizer'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                # logits=tf.reshape(self.prediction_logits, shape=[-1]),
                # labels=tf.reshape(self.activity_label, shape=[-1])))
                logits=self.prediction_logits,
                labels=self.activity_label))

            var_list = [var for var in tf.trainable_variables()]

            print(var_list)

            opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(self.cost, var_list)
            train_op = opt.apply_gradients(zip(grads, var_list))

            self.optimizer = train_op

        self.loss_summary = tf.summary.scalar('prediction loss', self.cost)
        self.accuracy_summary = tf.summary.scalar('prediction accuracy', self.accuracy)
        self.train_summary = tf.summary.merge([self.loss_summary, self.accuracy_summary])

        self.validation_loss_summary = tf.summary.scalar('pred validation loss', self.cost)
        self.validation_accuracy_summary = tf.summary.scalar('pred validation accuracy', self.accuracy)
        self.validation_summary = tf.summary.merge([self.validation_loss_summary,
                                                    self.validation_accuracy_summary])

        self.file_writer = tf.summary.FileWriter(self.log_folder)

        self.saver = tf.train.Saver()

    def train(self):
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()  # (log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            # self.file_writer.add_graph(sess.graph)

            if not self.is_data_loaded():
                self.load_data()

            for epoch in range(self.num_epochs):
                for i in range(0, len(self.train_inputs), self.batch_size):
                    inputs_batch = self.train_inputs[i: i + self.batch_size]
                    labels_batch = self.train_activity_labels[i: i + self.batch_size]

                    _, loss, accuracy, pred_output = sess.run(
                        [self.optimizer, self.cost, self.accuracy, self.prediction],
                        feed_dict={self.input: inputs_batch,
                                   self.activity_label: labels_batch})

                    print(i, ',', epoch)
                    print(loss)
                    print(accuracy)
                    print(np.argmax(pred_output, 1).tolist())
                    print(np.argmax(labels_batch, 1).tolist())
                    print('--------------------------------')

                    if i == 0:
                        self.file_writer.add_summary(
                            (sess.run(self.train_summary,
                                      feed_dict={self.input: inputs_batch,
                                                 self.activity_label: labels_batch}))
                            , epoch)

                        self.file_writer.add_summary(
                            (sess.run(self.validation_summary,
                                      feed_dict={self.input: self.test_inputs[:100],
                                                 self.activity_label: self.test_activity_labels[:100]}))
                            , epoch)

            loss, accuracy, pred_output = sess.run(
                [self.cost, self.accuracy, self.prediction],
                # feed_dict={self.input: self.test_inputs[100:],
                #            self.activity_label: self.test_activity_labels[100:]})
                feed_dict={self.input: self.test_inputs,
                           self.activity_label: self.test_activity_labels})
            print('test loss: ', loss)
            print('test accuracy: ', accuracy)

            # cm = ConfusionMatrix(actual_vector=self.test_activity_labels[100:],
            #                      predict_vector=pred_output)
            # print('Confusion Matrix:')
            # print(cm)

            print('--------------------------------')

            save_path = self.saver.save(sess, self.model_path)
            print("Survival model saved in file: %s" % save_path)

    @staticmethod
    def __length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def __last_relevant(output, length_):
        batch_size_ = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size_) * max_length + (length_ - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant
