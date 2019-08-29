import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from deep_model.deep_conv_lstm_classifier_complex_cnn import DeepConvLSTMClassifier
from preprocessing.pamap2_reader import normalized_pamap2_rnn_input_train_test


class CoTeaching:
    def __init__(self, config):
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.series_max_len = config.series_max_len

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
        self.learner_1_train_labels = train_activity_labels[0:(train_set_len / 2)]
        self.learner_2_train_labels = train_activity_labels[(train_set_len / 2):]

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
                for i in range(0, len(self.learner_1_train_labels), self.batch_size):
                    learner_1_inputs_batch = self.learner_1_train_inputs[i: i + self.batch_size]
                    learner_1_labels_batch = self.learner_1_train_labels[i: i + self.batch_size]
                    learner_2_inputs_batch = self.learner_2_train_inputs[i: i + self.batch_size]
                    learner_2_labels_batch = self.learner_2_train_labels[i: i + self.batch_size]

                    print(np.shape(learner_1_inputs_batch))
                    print(np.shape(learner_1_labels_batch))
                    print(np.shape(learner_2_inputs_batch))
                    print(np.shape(learner_2_labels_batch))

                    detailed_loss_1 = sess.run([self.learner_1.detailed_cost],
                                               feed_dict={self.learner_1.input: learner_1_inputs_batch,
                                                          self.learner_1.activity_label: learner_1_labels_batch})

                    learner1_min_loss_indices = np.argsort(detailed_loss_1)[0:int(len(detailed_loss_1)*remember_rate)]
                    learner1_min_loss_smaples = np.array([learner_1_inputs_batch for i in learner1_min_loss_indices])
                    learner1_min_loss_labels = np.array([learner_1_labels_batch for i in learner1_min_loss_indices])

                    detailed_loss_2 = sess.run([self.learner_2.detailed_cost],
                                               feed_dict={self.learner_2.input: learner_2_inputs_batch,
                                                          self.learner_2.activity_label: learner_2_labels_batch})

                    learner2_min_loss_indices = np.argsort(detailed_loss_2)[0:int(len(detailed_loss_2) * remember_rate)]
                    learner2_min_loss_smaples = np.array([learner_2_inputs_batch for i in learner2_min_loss_indices])
                    learner2_min_loss_labels = np.array([learner_2_labels_batch for i in learner2_min_loss_indices])

                    _, loss_1, accuracy_1 = sess.run(
                        [self.learner_1.optimizer, self.learner_1.cost, self.learner_1.accuracy],
                        feed_dict={self.learner_1.input: learner2_min_loss_smaples,
                                   self.learner_1.activity_label: learner2_min_loss_labels})

                    _, loss_2, accuracy_2 = sess.run(
                        [self.learner_2.optimizer, self.learner_2.cost, self.learner_2.accuracy],
                        feed_dict={self.learner_2.input: learner1_min_loss_smaples,
                                   self.learner_2.activity_label: learner1_min_loss_indices})

                    print(i, ',', epoch)
                    print('learner1 loss: ', loss_1)
                    print('learner1 acc: ', accuracy_1)
                    # print(np.argmax(pred_output, 1).tolist())
                    # print(np.argmax(labels_batch, 1).tolist())
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

                if epoch % 5 == 0:
                    remember_rate *= forget_rate

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

            # self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels, 1),
            #                              y_pred=np.argmax(pred_output, 1),
            #                              save_addr=self.log_folder + '/score_plots.png')
            #
            # self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels, 1),
            #                              y_pred=np.argmax(pred_output, 1),
            #                              save_addr=self.log_folder + '/score_plots_2.png', fig_size=[20, 20])
            #
            # self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels, 1),
            #                              y_pred=np.argmax(pred_output, 1),
            #                              save_addr=self.log_folder + '/score_plots_3.png', fig_size=[5, 5])
            #
            # self.__draw_pred_score_plots(y_true=np.argmax(self.test_activity_labels, 1),
            #                              y_pred=np.argmax(pred_output, 1),
            #                              save_addr=self.log_folder + '/score_plots_4.png', fig_size=[30, 30])
            #
            # self.__visualize_data(start=0, end=self.test_inputs.shape[0], predicted_labels=np.argmax(pred_output, 1),
            #                       test_data=True)

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

            # self.file_writer.add_summary(
            #     (sess.run(self.images_summary,
            #               feed_dict={self.input: self.test_inputs[:100],
            #                          self.activity_label: self.test_activity_labels[:100]}))
            #     , epoch)

            # save_path = self.saver.save(sess, self.model_path)
            # print("Survival model saved in file: %s" % save_path)

