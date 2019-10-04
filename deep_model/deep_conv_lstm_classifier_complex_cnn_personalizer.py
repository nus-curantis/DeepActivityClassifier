import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

from preprocessing.time_series_reader_and_visualizer import get_our_dataset_labels_names
from preprocessing.data_to_rnn_input_transformer import data_to_rnn_input_train_test, normalized_rnn_input_train_test
from preprocessing.wharf_reader import normalized_wharf_rnn_input_train_test
from preprocessing.pamap2_reader import pamap2_rnn_input_train_test, get_pamap_dataset_labels_names
# todo: also test normalized pamap input
# from preprocessing.pamap2_reader_flexible import pamap2_rnn_input_train_test

from clustering.segments_clusterer import ClusteringExecutor


class DeepConvLSTMClassifier:
    def __init__(self, config):

        # model parameters
        self.series_max_len = config.series_max_len
        self.rnn_hidden_units = config.rnn_hidden_units
        self.input_representations = config.input_representations
        self.num_classes = config.num_classes
        self.dropout_prob = config.dropout_prob

        self.filter_1_x = config.filter_1_x
        self.filter_1_y = config.filter_1_y
        self.filters_num_1 = config.filters_num_1
        self.stride_1_x = config.stride_1_x
        self.stride_1_y = config.stride_1_y
        self.filter_2_x = config.filter_2_x
        self.filter_2_y = config.filter_2_y
        self.filters_num_2 = config.filters_num_2
        self.stride_2_x = config.stride_2_x
        self.stride_2_y = config.stride_2_y
        self.filter_3_x = config.filter_3_x
        self.filter_3_y = config.filter_3_y
        self.filters_num_3 = config.filters_num_3
        self.stride_3_x = config.stride_3_x
        self.stride_3_y = config.stride_3_y

        # learning parameters
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.num_epochs_tuning = config.num_epochs_tuning
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
        self.conv_w_1 = None
        self.conv_b_1 = None
        self.conv_w_2 = None
        self.conv_b_2 = None
        self.conv_w_3 = None
        self.conv_b_3 = None
        self.cnn_layer_1_out = None
        self.cnn_layer_2_out = None
        self.cnn_layer_3_out = None
        self.rnn_cell = None
        self.rnn_output = None
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
        self.train_inputs_tuning = None  # inputs belonging to only one person or few people
        self.test_inputs = None
        self.test_inputs_tuning = None
        self.train_activity_labels = None
        self.train_activity_labels_tuning = None
        self.test_activity_labels = None
        self.test_activity_labels_tuning = None

        self.dataset_labels = None

    # def set_data(self,
    #              train_inputs,
    #              test_inputs,
    #              train_activity_labels,
    #              test_activity_labels):
    #     self.train_inputs = train_inputs
    #     self.test_inputs = test_inputs
    #     self.train_activity_labels = train_activity_labels
    #     self.test_activity_labels = test_activity_labels

    def is_data_loaded(self):
        for data in [
            self.train_inputs,
            self.test_inputs,
            self.train_activity_labels,
            self.test_activity_labels,
            self.train_inputs_tuning,
            self.train_activity_labels_tuning,
            self.test_inputs_tuning,
            self.test_activity_labels_tuning
        ]:
            if data is None:
                return False

        return True

    def load_data(self):
        self.train_inputs, self.test_inputs, self.train_activity_labels, self.test_activity_labels = \
            pamap2_rnn_input_train_test(target_dir='../dataset/PAMAP2_Dataset/split/1/',
                                        split_series_max_len=self.series_max_len)  # pamap2 dataset

        self.train_inputs_tuning, self.test_inputs_tuning, \
        self.train_activity_labels_tuning, self.test_activity_labels_tuning = \
            pamap2_rnn_input_train_test(target_dir='../dataset/PAMAP2_Dataset/split/2/',
                                        split_series_max_len=self.series_max_len)  # pamap2 dataset, data of one person

        self.dataset_labels = get_pamap_dataset_labels_names()

        print('len(self.train_inputs):', len(self.train_inputs))
        print('len(self.train_activity_labels):', len(self.train_activity_labels))
        print('len(self.train_inputs_tuning):', len(self.train_inputs_tuning))
        print('len(self.train_activity_labels_tuning):', len(self.train_activity_labels_tuning))
        print('len(self.test_inputs):', len(self.test_inputs))
        print('len(self.test_activity_labels):', len(self.test_activity_labels))
        print('len(self.test_inputs_tuning):', len(self.test_inputs_tuning))
        print('len(self.test_activity_labels_tuning):', len(self.test_activity_labels_tuning))

    def build_model(self):
        with tf.name_scope('cnn'):
            self.conv_w_1 = tf.Variable(tf.truncated_normal([self.filter_1_x, self.filter_1_y, 1, self.filters_num_1]))
            self.conv_b_1 = tf.Variable(tf.zeros([self.filters_num_1]))

            self.conv_w_2 = tf.Variable(tf.truncated_normal([self.filter_2_x, self.filter_2_y, 1, self.filters_num_2]))
            self.conv_b_2 = tf.Variable(tf.zeros([self.filters_num_2]))

            self.conv_w_3 = tf.Variable(tf.truncated_normal([self.filter_3_x, self.filter_3_y,
                                                             self.filters_num_2, self.filters_num_3]))
            self.conv_b_3 = tf.Variable(tf.zeros([self.filters_num_3]))

            expanded_input = tf.expand_dims(self.input, -1)
            self.cnn_layer_1_out = tf.nn.conv2d(expanded_input,
                                                filter=self.conv_w_1,
                                                strides=[1, self.stride_1_x, self.stride_1_y, 1],
                                                padding='VALID') + self.conv_b_1

            print('expanded_input: ', expanded_input)
            print('self.cnn_layer_1_out before reshape : ', self.cnn_layer_1_out)

            self.cnn_layer_1_out = tf.reshape(self.cnn_layer_1_out,
                                              shape=[-1,
                                                     self.cnn_layer_1_out.shape[1],
                                                     self.filters_num_1 * self.cnn_layer_1_out.shape[2], 1])

            self.cnn_layer_1_out = self.activation_function(batch_norm(self.cnn_layer_1_out))
            # todo: Is normalization correct?

            print('self.cnn_layer_1_out : ', self.cnn_layer_1_out)

            self.cnn_layer_2_out = tf.nn.conv2d(self.cnn_layer_1_out,
                                                filter=self.conv_w_2,
                                                strides=[1, self.stride_2_x, self.stride_2_y, 1],
                                                padding='VALID') + self.conv_b_2

            self.cnn_layer_2_out = self.activation_function(batch_norm(self.cnn_layer_2_out))

            print('self.cnn_layer_2_out : ', self.cnn_layer_2_out)

            self.cnn_layer_3_out = tf.nn.conv2d(self.cnn_layer_2_out,
                                                filter=self.conv_w_3,
                                                strides=[1, self.stride_3_x, self.stride_3_y, 1],
                                                padding='VALID') + self.conv_b_3

            print('self.cnn_layer_3_out before reshape : ', self.cnn_layer_3_out)

            self.cnn_layer_3_out = tf.reshape(self.cnn_layer_3_out,
                                              shape=[-1,
                                                     self.cnn_layer_3_out.shape[1],
                                                     self.filters_num_3 * self.cnn_layer_3_out.shape[2]])

            self.embedded_input = self.activation_function(batch_norm(self.cnn_layer_3_out))

            print('self.cnn_layer_3_out : ', self.embedded_input)

        with tf.name_scope('rnn'):
            # self.rnn_cell = rnn.GRUCell(num_units=self.rnn_hidden_units,
            #                             kernel_initializer=tf.orthogonal_initializer())

            self.rnn_cell = rnn.LSTMCell(num_units=self.rnn_hidden_units, name='cell_1')

            self.rnn_output, _ = tf.nn.dynamic_rnn(
                cell=self.rnn_cell, inputs=self.embedded_input,
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
            self.avg_pooling = tf.reduce_mean(self.rnn_output, axis=1)
            self.max_pooling = tf.reduce_max(self.rnn_output, axis=1)
            self.last_pooling = self.__last_relevant(self.rnn_output, self.__length(self.embedded_input))

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
                                      feed_dict={self.input: self.test_inputs_tuning[:100],
                                                 self.activity_label: self.test_activity_labels_tuning[:100]}))
                            , epoch)

            for epoch in range(self.num_epochs_tuning):
                for i in range(0, len(self.train_inputs_tuning), self.batch_size):
                    inputs_batch = self.train_inputs_tuning[i: i + self.batch_size]
                    labels_batch = self.train_activity_labels_tuning[i: i + self.batch_size]

                    _, loss, accuracy, pred_output = sess.run(
                        [self.optimizer, self.cost, self.accuracy, self.prediction],
                        feed_dict={self.input: inputs_batch,
                                   self.activity_label: labels_batch})

                    print(i, ',', epoch + self.num_epochs)
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
                            , epoch + self.num_epochs)

                        self.file_writer.add_summary(
                            (sess.run(self.validation_summary,
                                      feed_dict={self.input: self.test_inputs_tuning[:100],
                                                 self.activity_label: self.test_activity_labels_tuning[:100]}))
                            , epoch + self.num_epochs)

            save_path = self.saver.save(sess, self.model_path)
            print("Survival model saved in file: %s" % save_path)

    def test(self):

        config = tf.ConfigProto()  # (log_device_placement=True)
        config.gpu_options.allow_growth = True

        init = tf.global_variables_initializer()

        with tf.Session(config=config) as sess:
            sess.run(init)
            self.saver.restore(sess, self.model_path)

            if not self.is_data_loaded():
                self.load_data()

            loss, accuracy, pred_output = sess.run(
                [self.cost, self.accuracy, self.prediction],
                feed_dict={self.input: self.test_inputs,
                           self.activity_label: self.test_activity_labels})
            print('test loss on all data: ', loss)
            print('test accuracy on all data: ', accuracy)

            print(np.shape(pred_output))
            print(np.shape(self.test_activity_labels))

            print('test precision score on all data: ', precision_score(y_true=np.argmax(self.test_activity_labels, 1),
                                                                        y_pred=np.argmax(pred_output, 1), average=None))
            print('test recall score on all data: ', recall_score(y_true=np.argmax(self.test_activity_labels, 1),
                                                                  y_pred=np.argmax(pred_output, 1), average=None))

            print('test f1 score on all data: ', f1_score(y_true=np.argmax(self.test_activity_labels, 1),
                                                          y_pred=np.argmax(pred_output, 1), average=None))

            print('test confusion matrix on all data: ',
                  confusion_matrix(y_true=np.argmax(self.test_activity_labels, 1),
                                   y_pred=np.argmax(pred_output, 1)))

            print('           --------------            ')

            loss, accuracy, pred_output = sess.run(
                [self.cost, self.accuracy, self.prediction],
                feed_dict={self.input: self.test_inputs_tuning,
                           self.activity_label: self.test_activity_labels_tuning})
            print('test loss on target person: ', loss)
            print('test accuracy on target person: ', accuracy)

            print(np.shape(pred_output))
            print(np.shape(self.test_activity_labels))

            print('test precision score: ', precision_score(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                                            y_pred=np.argmax(pred_output, 1), average=None))
            print('test recall score: ', recall_score(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                                      y_pred=np.argmax(pred_output, 1), average=None))

            print('test f1 score: ', f1_score(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                              y_pred=np.argmax(pred_output, 1), average=None))

            print('test confusion matrix: ',
                  confusion_matrix(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                   y_pred=np.argmax(pred_output, 1)))

            self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                         y_pred=np.argmax(pred_output, 1),
                                         save_addr=self.log_folder + '/score_plots.png')

            self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                         y_pred=np.argmax(pred_output, 1),
                                         save_addr=self.log_folder + '/score_plots_2.png', fig_size=[20, 20])

            self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                         y_pred=np.argmax(pred_output, 1),
                                         save_addr=self.log_folder + '/score_plots_3.png', fig_size=[5, 5])

            self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels_tuning, 1),
                                         y_pred=np.argmax(pred_output, 1),
                                         save_addr=self.log_folder + '/score_plots_4.png', fig_size=[30, 30])

            print('-------------------------------------')

            # Checking performance of the model on different clusters of some of the activities:

            clustering_executor = ClusteringExecutor()
            clustering_executor.set_all_data(
                all_train_data=self.train_inputs,
                all_test_data=self.test_inputs,
                all_train_labels=self.train_activity_labels,
                all_test_labels=self.test_activity_labels
            )

            # for class_name in ['nordic_walking', 'running']:
            for class_name in ['cycling']:
                print('<<<<<<<<<<<<<<<<<<<< ' + class_name + ' >>>>>>>>>>>>>>>>>>>>>')

                num_clusters = 2  # 3
                num_segments = 200  # 300
                clustered_train_data, clustered_train_labels, train_cluster_nums, \
                    clustered_test_data, clustered_test_labels, test_cluster_nums = \
                    clustering_executor.get_clustered_data(class_name=class_name, num_segments=num_segments,
                                                           series_max_len=self.series_max_len, num_clusters=num_clusters
                                                           )

                clustered_train_data_indices = clustering_executor.selected_train_data_indices
                clustered_test_data_indices = clustering_executor.selected_test_data_indices

                pred_output_train = sess.run(
                    [self.prediction],
                    feed_dict={self.input: self.train_inputs,
                               self.activity_label: self.train_activity_labels})

                pred_output_test = sess.run(
                    [self.prediction],
                    feed_dict={self.input: self.test_inputs,
                               self.activity_label: self.test_activity_labels})

                pred_output_train = np.reshape(pred_output_train, newshape=[-1, np.array(pred_output_train).shape[-1]])
                pred_output_test = np.reshape(pred_output_test, newshape=[-1, np.array(pred_output_test).shape[-1]])

                for cluster_num in range(num_clusters):
                    train_data_indices = []

                    counter = 0
                    for _ in clustered_train_data:
                        if train_cluster_nums[counter] == cluster_num:
                            train_data_indices.append(clustered_train_data_indices[counter])

                        counter += 1

                    test_data_indices = []

                    counter = 0
                    for _ in clustered_test_data:
                        if test_cluster_nums[counter] == cluster_num:
                            test_data_indices.append(clustered_test_data_indices[counter])

                        counter += 1

                    cluster_pred_output_train = np.array([pred_output_train[i] for i in train_data_indices])
                    cluster_pred_output_test = np.array([pred_output_test[i] for i in test_data_indices])
                    cluster_labels_train = np.array([self.train_activity_labels[i] for i in train_data_indices])
                    cluster_labels_test = np.array([self.test_activity_labels[i] for i in test_data_indices])

                    # train_acc = np.mean(
                    #     np.cast(np.equal(
                    #         np.argmax(cluster_pred_output_train, 1), np.argmax(cluster_labels_train, 1)
                    #     ), np.float)
                    # )
                    #
                    # test_acc = np.mean(
                    #     np.cast(np.equal(
                    #         np.argmax(cluster_pred_output_test, 1), np.argmax(cluster_labels_test, 1)
                    #     ), np.float)
                    # )

                    train_acc = accuracy_score(y_true=np.argmax(cluster_labels_train, 1),
                                               y_pred=np.argmax(cluster_pred_output_train, 1))

                    test_acc = accuracy_score(y_true=np.argmax(cluster_labels_test, 1),
                                              y_pred=np.argmax(cluster_pred_output_test, 1))

                    print('train samples of the cluster: ', len(cluster_labels_train))
                    print('train accuracy on cluster ' + str(cluster_num) + ': ', train_acc)
                    print('test accuracy on cluster ' + str(cluster_num) + ': ', test_acc)

                    print('train precision score: ', precision_score(y_true=np.argmax(cluster_labels_train, 1),
                                                                     y_pred=np.argmax(cluster_pred_output_train, 1),
                                                                     average=None))
                    print('train recall score: ', recall_score(y_true=np.argmax(cluster_labels_train, 1),
                                                               y_pred=np.argmax(cluster_pred_output_train, 1),
                                                               average=None))

                    print('train f1 score: ', f1_score(y_true=np.argmax(cluster_labels_train, 1),
                                                       y_pred=np.argmax(cluster_pred_output_train, 1),
                                                       average=None))

                    print('test samples of the cluster: ', len(cluster_labels_test))
                    print('test accuracy on cluster ' + str(cluster_num) + ': ', test_acc)

                    print('test precision score: ', precision_score(y_true=np.argmax(cluster_labels_test, 1),
                                                                    y_pred=np.argmax(cluster_pred_output_test, 1),
                                                                    average=None))
                    print('test recall score: ', recall_score(y_true=np.argmax(cluster_labels_test, 1),
                                                              y_pred=np.argmax(cluster_pred_output_test, 1),
                                                              average=None))

                    print('test f1 score: ', f1_score(y_true=np.argmax(cluster_labels_test, 1),
                                                      y_pred=np.argmax(cluster_pred_output_test, 1), average=None))

                    print('=======================================')

    def __draw_pred_score_plots(self, y_true, y_pred, save_addr, fig_size=[8.27, 11.69]):
        precision = np.array([precision_score(y_true=y_true, y_pred=y_pred, average=None)])
        recall = np.array([recall_score(y_true=y_true, y_pred=y_pred, average=None)])
        f1 = np.array([f1_score(y_true=y_true, y_pred=y_pred, average=None)])
        confusion_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)

        plt.clf()
        fig, axs = plt.subplots(4, 1)
        fig.set_figheight(fig_size[1])
        fig.set_figwidth(fig_size[0])
        col_label = self.dataset_labels

        print('len:', len(col_label))
        print(precision.shape)
        print(confusion_mat.shape)

        axs[0].axis('tight')
        axs[0].axis('off')
        precision_table = axs[0].table(cellText=precision, colLabels=col_label, rowLabels=['precision'], loc='center')

        axs[1].axis('tight')
        axs[1].axis('off')
        recall_table = axs[1].table(cellText=recall, colLabels=col_label, rowLabels=['recall'], loc='center')

        axs[2].axis('tight')
        axs[2].axis('off')
        f1_table = axs[2].table(cellText=f1, colLabels=col_label, rowLabels=['f1 score'], loc='center')

        axs[3].axis('tight')
        axs[3].axis('off')
        confusion_table = axs[3].table(cellText=confusion_mat, colLabels=col_label, rowLabels=col_label, loc='center')

        plt.savefig(save_addr)

    @staticmethod
    def __length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def __last_relevant(output, length_):
        print('lest_rel')
        print(output)
        print(length_)

        batch_size_ = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size_) * max_length + (length_ - 1)
        flat = tf.reshape(output, [-1, out_size])

        print(flat)
        print(index)

        relevant = tf.gather(flat, index)
        return relevant
