# Word embedding evaluator

This script takes as input:
 - The file containing word embedding vectors 
 - The file containing the vocabulary of the word embedding
 - The text file on which the embedding will be evaluated on
 
It initializes an LSTM Language Model using the embedding vectors as the first NN layer.

It continues to train the Language Model on a series of 10-word sentences that help predict the 11th word (label).

Finaly it outputs the Score of the embedding based on the model's training loss.
