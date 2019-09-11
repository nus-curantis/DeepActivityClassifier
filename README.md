### Deep Models Explanations: 
*simple_deep_activity_classifier*: A simple rnn without any embedding_layer or pooling_layer
> rnn + fc (2 + 1 layers) 

*deep_activity_classifier_with_pooling*: simple rnn + pooling layer
> rnn + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers) 

*deep_activity_classifier_with_pooling_and_embedding*: embedding layer + simple rnn + pooling layer
> embedding matrix(3 * 5) + rnn + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers)

*deep_conv_lstm_classifier*: cnn + simple rnn + pooling layer
> cnn network(single conv layer -> 5 filters of dimension [6, 3, 1]) + rnn + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers)

*new deep_conv_lstm_classifier* cnn + simple rnn + time distributed layer (+ dropout) + pooling layer (+ dropout)
> cnn network(single conv layer -> 15 filters of dimension [30, 1, 1]) + rnn + time distributed layer (1 layer + dropout) + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers + dropout)

*(deep_conv_lstm_classifier_all_pooling)*
> A model to test what will happen if all rnn outputs get fed to fc

*deep_conv_lstm_classifier_separate_rnns* cnn + 3 * simple rnn + 3 * time distributed layer (+ dropout) + 3 * pooling layer (+ dropout)
> cnn network(single conv layer -> 15 filters of dimension [30, 1, 1]) + rnn + time distributed layer (1 layer + dropout) + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers + dropout)

*deep_conv_lstm_classifier_bi_dir_rnn* cnn + bi dir rnn + time distributed layer (+ dropout) + pooling layer (+ dropout)
> cnn network(single conv layer -> 15 filters of dimension [30, 1, 1]) + rnn + time distributed layer (1 layer + dropout) + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers + dropout)


## Major Findings

- Readings from wrist-based accelerometer data is very noisy and we need a method to deal with the noise. 
- Reducing sampling rate by half (36 -> 18 Hz) reduces the accuracy significantly for personalized train/testing data.
- Embedding the inputs using a CNN layer before passing to RNN improves the performance (since CNN captures some time dependencies in the time series)
- An activity might have multiple distributions, and we found that the network is able to learn the different distributions despite having not enough data in different distributions.  So the inaccuracy is caused by noise and not by multiple distributions.  The model is overfitting to the noise.
- No effect on time distributed / bi-directional (except worse overfitting)
- Shorter segments are better (2.5s is better than 5s and 10s)
- Co-teaching for noisy data (as described in the co-teaching) did not improve (test/train accuracy around 0.64).   Idea: need to filter out the noise in some way?  (clustering to detect outliers? informativeness of a sample (cf active learning)?)
- Overlapping segments do not improve
