import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
from parameters import *

# unziping from corpus
def unzip(sentences, sentence_tags):
	for tagged_sentence in tagged_sentences:
		sentence, tags = zip(*tagged_sentence)
		sentences.append(np.array(sentence))
		sentence_tags.append(np.array(tags))
	return sentences, sentence_tags

#custom accuracy score which ignores <PAD>
def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

# one hot to num which is then converted to their original word tags
def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

tagged_sentences = nltk.corpus.treebank.tagged_sents()
#let us print the words of first sentence and its corresponding tags 
print("Let us see words and associated tags of first sentence")
for x in tagged_sentences[0]:
	print(x[0]+" --> "+ x[1])
print("Total tagged sentences: ", len(tagged_sentences))
print("Total tagged words:", len(nltk.corpus.treebank.tagged_words()))

#extracting sentences and corresponding tags
sentences=[]
tags=[]
sentences, tags= unzip(sentences, tags)

# encoding sentences and tags
encoded_sentences, encoded_tags= [], []
for s in sentences:
	s_int = []
	for w in s:
		try:
			s_int.append(word2index[w.lower()])
		except KeyError:									#******#
			s_int.append(word2index['<OOV>'])
	encoded_sentences.append(s_int)

for s in tags:
    encoded_tags.append([tag2index[t] for t in s])

max_length= len(max(encoded_sentences, key=len))
print("Max word length of sentences: " + str(max_length))

#creating conversion dicts
unique_words = set()
unique_tags = set()
for s in sentences:
    for w in s:
        unique_words.add(w.lower())
 
for ts in tags:
    for t in ts:
        unique_tags.add(t)

print("Unique words: ", len(unique_words))
print("Unique tags: ", len(unique_tags))

word2index = {w: i + 2 for i, w in enumerate(list(unique_words))}
word2index['<PAD>'] = 0  # The special value used for padding
word2index['<OOV>'] = 1  # The special value used for Out of Vocab words
tag2index = {t: i + 1 for i, t in enumerate(list(unique_tags))}
tag2index['<PAD>'] = 0  # The special value used to padding

#padding sentences and tags to max length
encoded_sentences = pad_sequences(encoded_sentences, maxlen= max_length, padding='post')
encoded_tags = pad_sequences(encoded_tags, maxlen= max_length, padding='post')

train_sentences, test_sentences, train_tags, test_tags = train_test_split(encoded_sentences, encoded_tags, test_size=0.3)

# model
model = Sequential()
model.add(InputLayer(input_shape=(max_length, )))
model.add(Embedding(len(word2index), embedding_dim))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy', ignore_class_accuracy(0)])
model.summary()

model.fit(train_sentences, to_categorical(train_tags, len(tag2index)), batch_size= batch_size, epochs= epochs, validation_split= val_split)

#test accuracy
scores = model.evaluate(test_sentences, to_categorical(test_tags, len(tag2index)))
print("Acc: "+ str(scores[1] * 100))   # acc: 99.09751977804825

# Let's test our model on sample data
test_samples = [
    "running is very important for me .".split(),
    "I was running every day for a month .".split()
]

predictions = model.predict(test_samples_X)
print(predictions, predictions.shape)

test_results= logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
for x in range(len(test_results)):
	print(test_samples[x])
	print(test_results[x])
	print()