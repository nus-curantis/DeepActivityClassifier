import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm

import numpy as np

from sklearn.metrics import recall_score, precision_score, f1_score

from preprocessing.data_to_rnn_input_transformer import data_to_rnn_input
from preprocessing.time_series_reader_and_visualizer import num_to_activity


class DeepActivityClassifier:
    def __init__(self, config):

        # model parameters
        self.embedding_out_size = config.embedding_out_size
        self.series_max_len = config.series_max_len
        self.rnn_hidden_units = config.rnn_hidden_units
        self.input_representations = config.input_representations
        self.num_classes = config.num_classes

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
        self.embedded_input = None
        self.rnn_cell = None
        self.rnn_output = None
        self.avg_pooling = None
        self.max_pooling = None
        self.last_pooling = None
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
        self.validation_loss_summary = None
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
        self.train_inputs, self.train_activity_labels = data_to_rnn_input()

        # self.train_inputs = self.train_inputs[:, :, -1]  # todo: delete this test
        # self.train_inputs = np.reshape(self.train_inputs,
        #                                newshape=[self.train_inputs.shape[0], self.train_inputs.shape[1], 1])

    def build_model(self):
        with tf.name_scope('embedding'):  # for now, no embedding is used
            self.embedded_input = self.input

        with tf.name_scope('rnn'):
            # self.rnn_cell = rnn.GRUCell(num_units=self.rnn_hidden_units,
            #                             kernel_initializer=tf.orthogonal_initializer())

            self.rnn_cell = rnn.LSTMCell(num_units=self.rnn_hidden_units)

            self.rnn_output, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell, inputs=self.embedded_input,
                dtype=tf.float32, sequence_length=self.__length(self.embedded_input))

        with tf.name_scope('pooling'):
            # self.avg_pooling = tf.reduce_mean(self.rnn_output, axis=1)
            # self.max_pooling = tf.reduce_max(self.rnn_output, axis=1)
            # self.last_pooling = self.rnn_output[:, -1]
            self.last_pooling = self.__last_relevant(self.rnn_output, self.__length(self.embedded_input))

            # self.concatenated_poolings = tf.concat(
            #     [self.avg_pooling, self.max_pooling, self.last_pooling], axis=1
            # )

        with tf.name_scope('predictor'):
            self.dense_weights = {
                'first': tf.Variable(tf.truncated_normal(
                    # [3 * self.rnn_hidden_units, 2 * self.rnn_hidden_units])),
                    [self.rnn_hidden_units, 2 * self.rnn_hidden_units])),
                'second': tf.Variable(tf.truncated_normal(
                    [2 * self.rnn_hidden_units, 2 * self.rnn_hidden_units])),
                'third': tf.Variable(tf.truncated_normal([2 * self.rnn_hidden_units, self.num_classes]))
            }

            self.dense_biases = {
                'first': tf.Variable(tf.zeros([2 * self.rnn_hidden_units])),
                'second': tf.Variable(tf.zeros([2 * self.rnn_hidden_units])),
                'third': tf.Variable(tf.zeros([self.num_classes])),
            }

            self.hidden_layer_1 = batch_norm(tf.matmul(
                # self.concatenated_poolings, self.dense_weights['first']) + self.dense_biases['first'])
                self.last_pooling, self.dense_weights['first']) + self.dense_biases['first'])

            # self.hidden_layer_1 = tf.matmul(
            #     self.concatenated_poolings, self.dense_weights['first']) + self.dense_biases['first']

            self.hidden_layer_1 = self.activation_function(self.hidden_layer_1)

            self.hidden_layer_2 = batch_norm(tf.matmul(
                self.hidden_layer_1, self.dense_weights['second']) + self.dense_biases['second'])

            # self.hidden_layer_2 = tf.matmul(
            #     self.hidden_layer_1, self.dense_weights['second']) + self.dense_biases['second']

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
                # for i in [0]:
                    inputs_batch = self.train_inputs[i: i + self.batch_size]
                    labels_batch = self.train_activity_labels[i: i + self.batch_size]

                    # for labels in labels_batch:
                    #     print(labels.tolist())
                    # print('-------------------')

                    # for data in inputs_batch:
                    #     data = np.reshape(data, newshape=[self.input_representations, -1])
                    #     for axis in data:
                    #         print(axis.tolist())
                    #         print('-------')
                    #     print('===========')

                    _, loss, accuracy, pred_output = sess.run(
                        [self.optimizer, self.cost, self.accuracy, self.prediction],
                        feed_dict={self.input: inputs_batch,
                                   self.activity_label: labels_batch})

                    print(i, ',', epoch)
                    print(loss)
                    print(accuracy)
                    print(np.argmax(pred_output, 1).tolist())
                    print(np.argmax(labels_batch, 1).tolist())

                    # counter = 0
                    # for pred in pred_output:
                    #     try:
                    #         print(num_to_activity[np.argmax(labels_batch[counter])] + ' -->')
                    #     except KeyError:
                    #         print(str(np.argmax(labels_batch[counter])) + ' -->')
                    #
                    #     print(pred.tolist())
                    #     print(pred.shape)
                    #     print(labels_batch[counter].tolist())
                    #     print(labels_batch.shape)
                    #
                    #     counter += 1

                    print('--------------------------------')

                    if i == 0:
                        self.file_writer.add_summary(
                            (sess.run(self.loss_summary,
                                      feed_dict={self.input: inputs_batch,
                                                 self.activity_label: labels_batch}))
                            , epoch)

                        # self.file_writer.add_summary(
                        #     (sess.run(self.validation_loss_summary,
                        #               feed_dict={self.input: self.test_inputs[0: self.batch_size * 4],
                        #                          self.activity_label: self.test_activity_labels[0: self.batch_size * 4]}))
                        #     , epoch)

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


