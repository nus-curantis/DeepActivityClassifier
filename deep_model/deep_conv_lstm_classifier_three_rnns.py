import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm

import numpy as np

from sklearn.metrics import precision_score, recall_score

from preprocessing.data_to_rnn_input_transformer import data_to_rnn_input_train_test, normalized_rnn_input_train_test
from preprocessing.wharf_reader import normalized_wharf_rnn_input_train_test
from preprocessing.pamap2_reader import normalized_pamap2_rnn_input_train_test, pamap2_rnn_input_train_test


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
        self.embedded_input = None
        self.embedding = None
        self.conv_w = None
        self.conv_b = None
        self.rnn_cell_1 = None
        self.rnn_cell_2 = None
        self.rnn_cell_3 = None
        self.rnn_output_1 = None
        self.rnn_output_2 = None
        self.rnn_output_3 = None
        self.time_distributed_w = None
        self.time_distributed_b = None
        self.time_distributed_output = None
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
            pamap2_rnn_input_train_test(split_series_max_len=self.series_max_len)  # pamap2 dataset
            # normalized_rnn_input_train_test(data_path='../dataset/Chest_Accelerometer/data/',
            #                                 ignore_classes=[0, 2, 5, 6],
            #                                 split_series_max_len=self.series_max_len)  # chest dataset
            # data_to_rnn_input_train_test(data_path='../dataset/MHEALTHDATASET/', ignore_classes=[0, 12],
            #                              split_series_max_len=self.series_max_len)
            # data_to_rnn_input_train_test(
            #     split_series_max_len=self.series_max_len,
            #     ignore_classes=[1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])  # our dataset
            # data_to_rnn_input_train_test(split_series_max_len=self.series_max_len)  # our dataset
            # data_to_rnn_input_train_test(data_path='../dataset/MHEALTHDATASET/',
            #                              split_series_max_len=self.series_max_len)  # big dataset
            # normalized_rnn_input_train_test(data_path='../dataset/Chest_Accelerometer/data/',
            #                                 split_series_max_len=self.series_max_len)  # chest dataset
            # normalized_wharf_rnn_input_train_test(split_series_max_len=self.series_max_len)  # wahrf
            # data_to_rnn_input_train_test(data_path='../dataset/Chest_Accelerometer/data/')  # chest without normalizing

        print(len(self.train_inputs))
        print(len(self.train_activity_labels))
        print(len(self.test_inputs))
        print(len(self.test_activity_labels))

    def build_model(self):
        # with tf.name_scope('embedding'):
        #     # self.embedded_input = self.input
        #
        #     self.embedding = tf.Variable(tf.truncated_normal([self.input_representations, self.embedding_out_size]))
        #     # flattened_embedded_input = tf.nn.embedding_lookup(  # todo: solve the problem of this
        #     #     self.embedding, tf.reshape(self.input, shape=[-1, self.input_representations]))
        #     flattened_embedded_input = tf.matmul(
        #         tf.reshape(self.input, shape=[-1, self.input_representations]), self.embedding)
        #     self.embedded_input = tf.reshape(flattened_embedded_input,
        #                                      shape=[-1, self.series_max_len, self.embedding_out_size])

        # with tf.name_scope('cnn'):
        #     self.conv_w = tf.Variable(tf.truncated_normal([self.split_len, self.input_representations,
        #                                                   1, self.filters_num]))
        #     self.conv_b = tf.Variable(tf.zeros([self.filters_num]))
        #
        #     expanded_input = tf.expand_dims(self.input, -1)
        #     self.embedded_input = tf.nn.conv2d(expanded_input,
        #                                        filter=self.conv_w,
        #                                        strides=[1, self.split_len, 1, 1],
        #                                        # strides=[1, int(self.split_len / 2), 1, 1],
        #                                        padding='VALID') + self.conv_b
        #
        #     print('expanded_input: ', expanded_input)
        #     print('self.embedded_input : ', self.embedded_input)
        #
        #     self.embedded_input = tf.reshape(self.embedded_input,
        #                                      shape=[-1,
        #                                             self.embedded_input.shape[1] * self.embedded_input.shape[2],
        #                                             self.filters_num])
        #
        #     print('self.embedded_input : ', self.embedded_input)

        with tf.name_scope('cnn'):
            self.conv_w = tf.Variable(tf.truncated_normal([self.split_len, 1, 1, self.filters_num]))
            self.conv_b = tf.Variable(tf.zeros([self.filters_num]))

            expanded_input = tf.expand_dims(self.input, -1)
            self.embedded_input = tf.nn.conv2d(expanded_input,
                                               filter=self.conv_w,
                                               strides=[1, self.split_len, 1, 1],
                                               # strides=[1, int(self.split_len / 2), 1, 1],
                                               padding='VALID') + self.conv_b

            print('expanded_input: ', expanded_input)
            print('self.embedded_input : ', self.embedded_input)

            self.embedded_input = tf.reshape(self.embedded_input,
                                             shape=[-1,
                                                    self.embedded_input.shape[1] * self.embedded_input.shape[2],
                                                    self.filters_num])

            print('self.embedded_input : ', self.embedded_input)

        # with tf.name_scope('initial_dropout'):
        #     self.embedded_input = tf.nn.dropout(x=self.embedded_input, keep_prob=self.dropout_prob)

        with tf.name_scope('rnn_1'):
            # self.rnn_cell = rnn.GRUCell(num_units=self.rnn_hidden_units,
            #                             kernel_initializer=tf.orthogonal_initializer())

            self.rnn_cell_1 = rnn.LSTMCell(num_units=self.rnn_hidden_units, name='cell_1')

            self.rnn_output_1, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell_1, inputs=self.embedded_input,
                dtype=tf.float32, sequence_length=self.__length(self.embedded_input))

        with tf.name_scope('rnn_2'):
            self.rnn_cell_2 = rnn.LSTMCell(num_units=self.rnn_hidden_units, name='cell_2')

            self.rnn_output_2, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell_2, inputs=self.rnn_output_1,
                dtype=tf.float32, sequence_length=self.__length(self.embedded_input))

        with tf.name_scope('rnn_3'):
            self.rnn_cell_3 = rnn.LSTMCell(num_units=self.rnn_hidden_units, name='cell_3')

            self.rnn_output_3, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell_3, inputs=self.rnn_output_2,
                dtype=tf.float32, sequence_length=self.__length(self.embedded_input))

        # with tf.name_scope('time_distributed_layer'):
        #     rnn_output_reshaped = tf.reshape(self.rnn_output_3, shape=[-1, self.rnn_hidden_units])
        #
        #     self.time_distributed_w = tf.Variable(tf.truncated_normal([self.rnn_hidden_units, self.rnn_hidden_units]))
        #     self.time_distributed_b = tf.Variable(tf.zeros([self.rnn_hidden_units]))
        #
        #     time_distributed_output = tf.matmul(rnn_output_reshaped, self.time_distributed_w) + self.time_distributed_b
        #     self.time_distributed_output = tf.reshape(time_distributed_output, shape=tf.shape(self.rnn_output_1))
        #
        #     # test:
        #     self.time_distributed_output = tf.nn.dropout(self.time_distributed_output, self.dropout_prob)

        with tf.name_scope('pooling'):
            self.avg_pooling = tf.reduce_mean(self.rnn_output_3, axis=1)
            self.max_pooling = tf.reduce_max(self.rnn_output_3, axis=1)
            self.last_pooling = self.__last_relevant(self.rnn_output_3, self.__length(self.embedded_input))

            # self.avg_pooling = tf.reduce_mean(self.time_distributed_output, axis=1)
            # self.max_pooling = tf.reduce_max(self.time_distributed_output, axis=1)
            # self.last_pooling = self.__last_relevant(self.time_distributed_output, self.__length(self.embedded_input))

            self.concatenated_poolings = tf.concat(
                [self.avg_pooling, self.max_pooling, self.last_pooling], axis=1
            )

        with tf.name_scope('predictor'):
            self.dense_weights = {
                'first': tf.Variable(tf.truncated_normal(
                    [3 * self.rnn_hidden_units, 2 * self.rnn_hidden_units])),
                # [self.rnn_hidden_units, 2 * self.rnn_hidden_units])),
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
                self.concatenated_poolings, self.dense_weights['first']) + self.dense_biases['first'])
            # self.last_pooling, self.dense_weights['first']) + self.dense_biases['first'])

            # self.hidden_layer_1 = tf.matmul(
            #     self.concatenated_poolings, self.dense_weights['first']) + self.dense_biases['first']

            self.hidden_layer_1 = self.activation_function(self.hidden_layer_1)

            # test:
            self.hidden_layer_1 = tf.nn.dropout(self.hidden_layer_1, self.dropout_prob)

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

            # opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
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

        # self.conv_filters_summary = tf.summary.image(
        #     'conv filters', tf.reshape(self.conv_w, shape=[self.filters_num, self.split_len, 1, 1]))
        # self.conv_output_summary = tf.summary.image('conv outputs', tf.expand_dims(self.embedded_input, -1),
        #                                             max_outputs=30)
        # self.avg_pooling_summary = tf.summary.image(
        #     'avg pooling', tf.reshape(self.avg_pooling, shape=[-1, 2 * self.rnn_hidden_units, 1, 1]), max_outputs=30)
        # self.max_pooling_summary = tf.summary.image(
        #     'max pooling', tf.reshape(self.max_pooling, shape=[-1, 2 * self.rnn_hidden_units, 1, 1]), max_outputs=30)
        # self.last_pooling_summary = tf.summary.image(
        #     'mean pooling', tf.reshape(self.last_pooling, shape=[-1, 2 * self.rnn_hidden_units, 1, 1]), max_outputs=30)
        #
        # self.images_summary = tf.summary.merge([self.conv_filters_summary, self.conv_output_summary,
        #                                         self.avg_pooling_summary, self.max_pooling_summary,
        #                                         self.last_pooling_summary])

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

            print(np.shape(pred_output))
            print(np.shape(self.test_activity_labels))

            print('test precision score: ', precision_score(y_true=np.argmax(self.test_activity_labels, 1),
                                                            y_pred=np.argmax(pred_output, 1), average=None))
            print('test recall score: ', recall_score(y_true=np.argmax(self.test_activity_labels, 1),
                                                      y_pred=np.argmax(pred_output, 1), average=None))

            print('--------------------------------')

            # self.file_writer.add_summary(
            #     (sess.run(self.images_summary,
            #               feed_dict={self.input: self.test_inputs[:100],
            #                          self.activity_label: self.test_activity_labels[:100]}))
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
