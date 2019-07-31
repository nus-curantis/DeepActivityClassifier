# from preprocessing.data_to_rnn_input_transformer import data_to_rnn_input_train_test, normalized_rnn_input_train_test
# from preprocessing.wharf_reader import normalized_wharf_rnn_input_train_test
# # from preprocessing.pamap2_reader import normalized_pamap2_rnn_input_train_test, pamap2_rnn_input_train_test
# from preprocessing.pamap2_reader_flexible import pamap2_rnn_input_train_test
#
# class Dataset:
#     def __init__(self, name, normalized=False):
#         self.name = name
#         self.normalized = normalized
#
#     def load_data(self, series_max_len):
#         loader_func_map = {
#             'ours': self.__load_pamap_data(series_max_len)
#         }
#
#     def __load_pamap_data(self, series_max_len):
#         return pamap2_rnn_input_train_test(split_series_max_len=series_max_len)
#
#     def __load_ours_data(self):
#         pass
