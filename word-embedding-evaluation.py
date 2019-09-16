import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell, static_rnn
from nltk.corpus import brown
from operator import itemgetter
from keras.layers import Embedding, Dense, LSTM, Input
from keras.models import Sequential, Model
from keras.utils import to_categorical
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.tools import inspect_checkpoint as chkp
import math
import csv
import nltk
from keras.layers import Embedding

DIMENSIONS = -1 # defined by the vector-list.tsv dimensions (participant's choice)
WORD_LIST_NUM = 5010 # the fixed number of the words the participants should create embedding vectors for. the fixed word list (5010.txt) can be downloaded from the Data tab in https://race.nbg.gr/challenge-details/wordembedding 
TRAINING_EPOCHS = 200 # the number of LM model training epochs

# this is to allow to increase the 5010 vocabulary representation in the evaluation text, our evaluation text will have ~ 50% representation
# it is a threshold of consecutive unknown tokens in the evaluation text that trigger a SKIP_UNKNOWN_TOKENS word skip to avoid braking the in-sentence narrative
SKIP_UNKNOWN_TOKENS = 4 

vector_list_path = "vector-list.tsv"
word_list_path = "5010.txt"
evaluation_text_path = "evaluate.txt"

embeddings_index = {}
coefs = []
word_index = {}

# open the file containing the embedding vectors, this is the main deliverable and should be placed in the root of your repository
i=0
with open(vector_list_path,'r') as f:
    for line in f:
        values = line.replace('[','').replace(']','').replace(', ',' ').split()
        coefs.append(np.asarray(values[0:], dtype='float32'))
        i+=1

# check if the number of vectors in the vector_list_path == WORD_LIST_NUM
if WORD_LIST_NUM != i:
    raise ValueError("NBG-WEC101 - Word count is: ", i, ", expected: ", WORD_LIST_NUM);

DIMENSIONS = len(np.asarray(values[0:], dtype='float32'))

# open the fixed list of words as found in the 5010.txt file
i=0
with open(word_list_path, 'r') as f:
    for line in f:
        values = line.split()
        word = values[0].lower()
        embeddings_index[word] = coefs[i]
        word_index[line.replace('\n','').replace(' ','').lower()] = i
        i+=1

# initialize the embedding layer        
embedding_matrix = np.zeros((WORD_LIST_NUM + 1, DIMENSIONS))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word.lower())
    
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(WORD_LIST_NUM + 1,
                            DIMENSIONS,
                            weights=[embedding_matrix],
                            trainable=False)

# read the evaluation corpus
with open(evaluation_text_path,'r') as f:
    contents=f.read()

indexed_evaluation_text = []
complementary_word_index = {}
dots_index = []
unknown_index = []

index = 0
represented = 0
notrepresented = 0
corpus_pos = 0
vocab_pos = 0
end_of_sentence = 0
unknown_previous = 0
unknown_start = 0
unknown_end = 0

# create the indexed_evaluation_text that will be used for training
for word in contents.split():
    word = word.replace(',','')
    if word.endswith('.'):
        word = word[:-1]
        end_of_sentence = 1
    
    # if the word is a known token 
    try: 
        index = word_index[word.lower()]
        indexed_evaluation_text.append(index)
        represented += 1
        
        # keep an index of the end of sentences as we need to avoid training to predict the first of a new sentence
        if end_of_sentence == 1 and vocab_pos > 10: 
            dots_index.append(vocab_pos + 0.5)
            end_of_sentence = 0            
        unknown_previous = 0
        vocab_pos += 1

    # if the word is an unknown token
    except KeyError:
        notrepresented += 1
        
        # skip SKIP_UNKNOWN_TOKENS consecutive words
        if unknown_previous == 1:
            unknown_end += 1
            if unknown_end - unknown_start == SKIP_UNKNOWN_TOKENS:
                notrepresented -= SKIP_UNKNOWN_TOKENS
                unknown_index.append([unknown_start, unknown_end])
                unknown_start = corpus_pos
                unknown_end = corpus_pos
        else:
            unknown_start = corpus_pos
            unknown_end = corpus_pos
        unknown_previous = 1
    corpus_pos += 1

print("Vocabulary representation:", represented/notrepresented)    
    
input_layer=Input(shape=(1,))

embedding_text=embedding_layer(input_layer)

embedding_model=Model(inputs=input_layer, outputs=embedding_text)

sentences = []
next_word = []
test_data = indexed_evaluation_text
j = 0
k = 0

# construct the training data with sentences of 10 words that predict the 11th word
for i in range(0,len(test_data)-10, 1):
    
    # if next_word is the beginning of new sentence, skip
    if i+9 < dots_index[j] and i+10 > dots_index[j] and j < len(dots_index)-1: 
        j += 1
        continue
        
    # if sentence contains SKIP_UNKNOWN_TOKENS, skip
    if i >= unknown_index[k][0] and i+9 >= unknown_index[k][1]: 
        k += 1
        continue
    
    sentences.append(test_data[i:i+10])
    next_word.append(test_data[i+10])

print ("#Sentences:",len(sentences))
    
# create the dataset X and the corresponding labels to train the Keras implementation
X=np.zeros((len(sentences),10,DIMENSIONS))
y=np.zeros((len(sentences),1))

for i in range(len(sentences)):
    np_sentence=np.array(sentences[i])
    np_sentence.reshape(1,10)
    look_up=embedding_model.predict(np_sentence)
    X[i,:,:]=look_up.reshape(10,DIMENSIONS)
    y[i]=next_word[i]
    
y=to_categorical(y, WORD_LIST_NUM + 1)

# create a Keras sequential model
model=Sequential()
model.add(LSTM(DIMENSIONS, input_shape=(10,DIMENSIONS)))
model.add(Dense(WORD_LIST_NUM + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

history = model.fit(X, y,
          batch_size=100,
          epochs=TRAINING_EPOCHS)

# the 100-top-bounded score based on min loss
print("Score: ", 100 - min(history.history['loss'])*100)
