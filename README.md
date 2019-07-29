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


