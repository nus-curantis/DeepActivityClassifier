### Deep Models Explanations: 
*simple_deep_activity_classifier*: A simple rnn without any embedding_layer or pooling_layer
> rnn + fc (2 + 1 layers) 

*deep_activity_classifier_with_pooling*: simple rnn + pooling layer
> rnn + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers) 

*deep_activity_classifier_with_pooling_and_embedding*: embedding layer + simple rnn + pooling layer
> embedding matrix(3 * 5) + rnn + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers)

*deep_conv_lstm_classifier*: cnn + simple rnn + pooling layer
> cnn network(single conv layer -> 5 filters of dimension [6, 3, 1]) + rnn + pooling(last pooling + max pooling + mean pooling) + fc (2 + 1 layers)