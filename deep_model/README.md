# Simple RNN (4.2.1)

- implemented in `models_archive/simple_deep_activity_classifier.py`
- for running the code and setting configurations, DeepActivityClassifier class (of the abovementioned source code) should be imported in `main_classifier.py`  

# RNN + Embedding Layer (4.2.2)

- implemented in `models_archive/deep_activity_classifier_with_pooling_and_embedding.py`
- for running the code and setting configurations, DeepActivityClassifier class (of the abovementioned source code) should be imported in `main_classifier.py`
- Another version of this code is implemented at `models_archive/deep_conv_lstm_classifier_all_pooling.py`: This code tests the effect of using all RNN cell's outputs instead of using a pooling layer. 
In this code all the outputs of different cells of RNN is being fed to the fully connected layer. 
Using such a large output for fc layer is not practical. We only wanted to test how better the model will perform in this case to check if we should search for a better construction instead of pooling layer.
The accuracy got only slightly better when all the outputs were used and we deduced that there's no need to change the pooling layer. (Pooling layer improves accuracy when nothing is used and when all the outputs are used the accuracy won't get better significantly.) 

# Convolutional LSTM Network (4.2.3, 4.2.4, 4.2.5)

- implemented in `deep_conv_lstm_classifier.py`
- for running the code and setting configurations, DeepActivityClassifier class (of the abovementioned source code) should be imported in `main_classifier.py`
- A version of this model with bidirectional RNN is implemented in `deep_conv_lstm_classifier_bi_dir_rnn.py`. It should be used and configured in the same abovementioned way.

# Conv LSTM network with 3 RNNs 4.2.6
- implemented in `models_archive/deep_conv_lstm_classifier_seperate_rnns.py`
- In this model z, y and z axis are fed to three separate RNNs. The performance is not different when concatenated input of all 3 axises is fed to one RNN and hence we didn't develope this model any further.
- for running the code and setting configurations, DeepActivityClassifier class (of the abovementioned source code) should be imported in `main_classifier.py` 

# Complex Conv LSTM Network (4.2.7)
- implemented in `deep_conv_lstm_classifier_complex.py`
- for running the code and setting configurations, please use `main_classifier_complex_cnn.py`

## Personalized Complex Conv LSTM Network (4.2.7.1)
- implemented in `deep_conv_lstm_classifier_complex_personalizer.py`
- for running the code and setting configurations, please use `main_classifier_complex_cnn_personalize.py`

## Complex Conv LSTM Network (4.2.7) using all 52 dimensions of PAMAP2 dataset
 - implemented in `deep_conv_lstm_classifier_complex_52.py`
 - for running the code and setting configurations, please use `main_classifier_complex_cnn_52.py`
 - Another version which uses 4 convolution layers is implemented in `deep_conv_lstm_classifier_complex_52_4conv.py`
 - `4 conv` version has a better accuracy
 - for running `4 conv` version and setting configurations, please use `main_classifier_complex_cnn_52_4d.py`
 
# Collaboration of two Complex Conv LSTM Models in a CoTeaching network (4.2.8)
- implemented in `deep_colab_conv_lstm_complex_cnn.py`
- for running the code and setting configurations, please use `main_colab_classifiers_complex_cnn.py` 
 