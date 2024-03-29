import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

from deep_model.deep_conv_lstm_classifier_complex_cnn import DeepConvLSTMClassifier
from preprocessing.pamap2_reader import normalized_pamap2_rnn_input_train_test

from clustering.segments_clusterer import ClusteringExecutor


class CoTeaching:
    def __init__(self, config):
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.series_max_len = config.series_max_len
        self.model_path = config.model_path

        log_folder = config.log_folder
        config.log_folder += '__1'
        self.learner_1 = DeepConvLSTMClassifier(config=config, model_name='model_1__')
        config.log_folder = log_folder + '__2'
        self.learner_2 = DeepConvLSTMClassifier(config=config, model_name='model_2__')
        self.learner_1.build_model()
        self.learner_2.build_model()

        self.learner_1_train_inputs = None
        self.learner_1_train_labels = None
        self.learner_2_train_inputs = None
        self.learner_2_train_labels = None
        self.test_inputs = None
        self.test_labels = None

    def load_data(self):
        train_inputs, self.test_inputs, train_activity_labels, self.test_labels = \
            normalized_pamap2_rnn_input_train_test(split_series_max_len=self.series_max_len)  # pamap2 dataset

        train_set_len = len(train_inputs)

        self.learner_1_train_inputs = train_inputs[0:int(train_set_len/2)]
        self.learner_2_train_inputs = train_inputs[int(train_set_len/2):]
        self.learner_1_train_labels = train_activity_labels[0:int(train_set_len / 2)]
        self.learner_2_train_labels = train_activity_labels[int(train_set_len / 2):]

    def train_two_networks(self, forget_rate=0.2):
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()  # (log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            # self.file_writer.add_graph(sess.graph)

            self.load_data()

            remember_rate = 1
            for epoch in range(self.num_epochs):
                for i in range(0, len(self.learner_1_train_labels) - self.batch_size, self.batch_size):
                    learner_1_inputs_batch = self.learner_1_train_inputs[i: i + self.batch_size]
                    learner_1_labels_batch = self.learner_1_train_labels[i: i + self.batch_size]
                    learner_2_inputs_batch = self.learner_2_train_inputs[i: i + self.batch_size]
                    learner_2_labels_batch = self.learner_2_train_labels[i: i + self.batch_size]

                    detailed_loss_1 = sess.run([self.learner_1.detailed_cost],
                                               feed_dict={self.learner_1.input: learner_1_inputs_batch,
                                                          self.learner_1.activity_label: learner_1_labels_batch})
                    detailed_loss_1 = np.array(detailed_loss_1)

                    learner1_min_loss_indices = np.argsort(detailed_loss_1)[0:int(np.floor(detailed_loss_1.shape[1]*
                                                                                           remember_rate))]
                    learner1_min_loss_samples = np.array([learner_1_inputs_batch for i in learner1_min_loss_indices])
                    learner1_samples_shape = np.shape(learner1_min_loss_samples)

                    learner1_min_loss_samples = np.reshape(learner1_min_loss_samples,
                                                           newshape=[learner1_samples_shape[1],
                                                                     learner1_samples_shape[2],
                                                                     learner1_samples_shape[3]])
                    learner1_min_loss_labels = np.array([learner_1_labels_batch for i in learner1_min_loss_indices])
                    learner1_min_loss_labels = np.reshape(learner1_min_loss_labels,
                                                          newshape=[-1, np.shape(learner1_min_loss_labels)[-1]])

                    detailed_loss_2 = sess.run([self.learner_2.detailed_cost],
                                               feed_dict={self.learner_2.input: learner_2_inputs_batch,
                                                          self.learner_2.activity_label: learner_2_labels_batch})
                    detailed_loss_2 = np.array(detailed_loss_2)

                    learner2_min_loss_indices = np.argsort(detailed_loss_2)[0:int(np.floor(detailed_loss_2.shape[1]*
                                                                                           remember_rate))]
                    learner2_min_loss_samples = np.array([learner_2_inputs_batch for i in learner2_min_loss_indices])
                    learner2_samples_shape = np.shape(learner2_min_loss_samples)
                    learner2_min_loss_samples = np.reshape(learner2_min_loss_samples,
                                                           newshape=[learner2_samples_shape[1],
                                                                     learner2_samples_shape[2],
                                                                     learner2_samples_shape[3]])
                    learner2_min_loss_labels = np.array([learner_2_labels_batch for i in learner2_min_loss_indices])
                    learner2_min_loss_labels = np.reshape(learner2_min_loss_labels,
                                                          newshape=[-1, np.shape(learner2_min_loss_labels)[-1]])

                    _, loss_1, accuracy_1 = sess.run(
                        [self.learner_1.optimizer, self.learner_1.cost, self.learner_1.accuracy],
                        feed_dict={self.learner_1.input: learner2_min_loss_samples,
                                   self.learner_1.activity_label: learner2_min_loss_labels})

                    _, loss_2, accuracy_2 = sess.run(
                        [self.learner_2.optimizer, self.learner_2.cost, self.learner_2.accuracy],
                        feed_dict={self.learner_2.input: learner1_min_loss_samples,
                                   self.learner_2.activity_label: learner1_min_loss_labels})

                    print(i, ',', epoch)
                    print('learner1 loss: ', loss_1)
                    print('learner1 acc: ', accuracy_1)
                    print('learner2 loss: ', loss_2)
                    print('learner2 acc: ', accuracy_2)
                    print('--------------------------------')

                    if i == 0:
                        self.learner_1.file_writer.add_summary(
                            (sess.run(self.learner_1.train_summary,
                                      feed_dict={self.learner_1.input: learner_1_inputs_batch,
                                                 self.learner_1.activity_label: learner_1_labels_batch}))
                            , epoch)

                        self.learner_1.file_writer.add_summary(
                            (sess.run(self.learner_1.validation_summary,
                                      feed_dict={self.learner_1.input: self.test_inputs[:100],
                                                 self.learner_1.activity_label: self.test_labels[:100]}))
                            , epoch)

                        self.learner_2.file_writer.add_summary(
                            (sess.run(self.learner_2.train_summary,
                                      feed_dict={self.learner_2.input: learner_2_inputs_batch,
                                                 self.learner_2.activity_label: learner_2_labels_batch}))
                            , epoch)

                        self.learner_2.file_writer.add_summary(
                            (sess.run(self.learner_2.validation_summary,
                                      feed_dict={self.learner_2.input: self.test_inputs[:100],
                                                 self.learner_2.activity_label: self.test_labels[:100]}))
                            , epoch)

                if epoch % 5 == 1:
                    remember_rate *= 1 - forget_rate

            save_path = self.learner_1.saver.save(sess, self.model_path)
            print("First model saved in file: %s" % save_path)

            loss_1, accuracy_1, pred_output = sess.run(
                [self.learner_1.cost, self.learner_1.accuracy, self.learner_1.prediction],
                feed_dict={self.learner_1.input: self.test_inputs,
                           self.learner_1.activity_label: self.test_labels})
            print('learner_1 test loss: ', loss_1)
            print('learner_1 test accuracy: ', accuracy_1)

            print(np.shape(pred_output))
            print(np.shape(self.test_labels))

            print('learner_1 test precision score: ', precision_score(y_true=np.argmax(self.test_labels, 1),
                                                                      y_pred=np.argmax(pred_output, 1), average=None))
            print('learner_1 test recall score: ', recall_score(y_true=np.argmax(self.test_labels, 1),
                                                                y_pred=np.argmax(pred_output, 1), average=None))

            print('learner_1 test f1 score: ', f1_score(y_true=np.argmax(self.test_labels, 1),
                                                        y_pred=np.argmax(pred_output, 1), average=None))

            print('learner_1 test confusion matrix: ', confusion_matrix(y_true=np.argmax(self.test_labels, 1),
                                                                        y_pred=np.argmax(pred_output, 1)))

            print('--------------------------------')

            loss_2, accuracy_2, pred_output = sess.run(
                [self.learner_2.cost, self.learner_2.accuracy, self.learner_2.prediction],
                feed_dict={self.learner_2.input: self.test_inputs,
                           self.learner_2.activity_label: self.test_labels})
            print('learner_2 test loss: ', loss_2)
            print('learner_2 test accuracy: ', accuracy_2)

            print(np.shape(pred_output))
            print(np.shape(self.test_labels))

            print('learner_2 test precision score: ', precision_score(y_true=np.argmax(self.test_labels, 1),
                                                                      y_pred=np.argmax(pred_output, 1), average=None))
            print('learner_2 test recall score: ', recall_score(y_true=np.argmax(self.test_labels, 1),
                                                                y_pred=np.argmax(pred_output, 1), average=None))

            print('learner_2 test f1 score: ', f1_score(y_true=np.argmax(self.test_labels, 1),
                                                        y_pred=np.argmax(pred_output, 1), average=None))

            print('learner_2 test confusion matrix: ', confusion_matrix(y_true=np.argmax(self.test_labels, 1),
                                                                        y_pred=np.argmax(pred_output, 1)))

            print('================================')

    def test(self):  # tests the first network
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()  # (log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            self.learner_1.saver.restore(sess, self.model_path)

            if not self.learner_1.is_data_loaded():
                self.load_data()

            loss_1, accuracy_1, pred_output = sess.run(
                [self.learner_1.cost, self.learner_1.accuracy, self.learner_1.prediction],
                feed_dict={self.learner_1.input: self.test_inputs,
                           self.learner_1.activity_label: self.test_labels})
            print('learner_1 test loss: ', loss_1)
            print('learner_1 test accuracy: ', accuracy_1)

            print(np.shape(pred_output))
            print(np.shape(self.test_labels))

            print('learner_1 test precision score: ', precision_score(y_true=np.argmax(self.test_labels, 1),
                                                                      y_pred=np.argmax(pred_output, 1), average=None))
            print('learner_1 test recall score: ', recall_score(y_true=np.argmax(self.test_labels, 1),
                                                                y_pred=np.argmax(pred_output, 1), average=None))

            print('learner_1 test f1 score: ', f1_score(y_true=np.argmax(self.test_labels, 1),
                                                        y_pred=np.argmax(pred_output, 1), average=None))

            print('learner_1 test confusion matrix: ', confusion_matrix(y_true=np.argmax(self.test_labels, 1),
                                                                        y_pred=np.argmax(pred_output, 1)))

            print('--------------------------------')

            clustering_executor = ClusteringExecutor()
            clustering_executor.set_all_data(
                all_train_data=self.learner_1_train_inputs,
                all_test_data=self.test_inputs,
                all_train_labels=self.learner_1_train_labels,
                all_test_labels=self.test_labels
            )

            # The following code tests performance of the model on different clusters of some selected activities

            for class_name in ['nordic_walking', 'running']:
            # for class_name in ['cycling']:
                print('<<<<<<<<<<<<<<<<<<<< ' + class_name + ' >>>>>>>>>>>>>>>>>>>>>')

                num_clusters = 3
                num_segments = 300
                clustered_train_data, clustered_train_labels, train_cluster_nums, \
                    clustered_test_data, clustered_test_labels, test_cluster_nums = \
                    clustering_executor.get_clustered_data(class_name=class_name, num_segments=num_segments,
                                                           series_max_len=self.series_max_len, num_clusters=num_clusters
                                                           )

                clustered_train_data_indices = clustering_executor.selected_train_data_indices
                clustered_test_data_indices = clustering_executor.selected_test_data_indices

                pred_output_train = sess.run(
                    [self.learner_1.prediction],
                    feed_dict={self.learner_1.input: self.learner_1_train_inputs,
                               self.learner_1.activity_label: self.learner_1_train_labels})

                pred_output_test = sess.run(
                    [self.learner_1.prediction],
                    feed_dict={self.learner_1.input: self.test_inputs,
                               self.learner_1.activity_label: self.test_labels})

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

                    print('np.shape(pred_output_train)', np.shape(pred_output_train))
                    print('np.shape(pred_output_test)', np.shape(pred_output_test))
                    print('np.shape(self.train_activity_labels)', np.shape(self.learner_1_train_labels))
                    print('np.shape(self.test_activity_labels)', np.shape(self.test_labels))
                    print('np.shape(train_data_indices)', np.shape(train_data_indices))
                    print('np.shape(test_data_indices)', np.shape(test_data_indices))

                    cluster_pred_output_train = np.array([pred_output_train[i] for i in train_data_indices])
                    cluster_pred_output_test = np.array([pred_output_test[i] for i in test_data_indices])
                    cluster_labels_train = np.array([self.learner_1.train_activity_labels[i] for i in train_data_indices])
                    cluster_labels_test = np.array([self.learner_1.test_activity_labels[i] for i in test_data_indices])

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


