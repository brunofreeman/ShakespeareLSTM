import tensorflow as tf
import numpy as np
import os
import re
import datetime

MIN_COUNT = 50

file_count = 0
text = ""
for file in os.listdir("./shakespeare_data"):
    if file.endswith(".txt"):
        file_count += 1
        text += open(os.path.join("./shakespeare_data", file)).read().lower()

word_regex = "(?:[A-Za-z\']*(?:(?<!-)-(?!-))*[A-Za-z\']+)+"
punct_regex = "|\.|\?|!|,|;|:|-|\(|\)|\[|\]|\{|\}|\'|\"|\\|\/|<|>| |\t|\n"
regex = word_regex + punct_regex
words = re.findall(regex, text)
word_counts = dict()

for word in words: #create a dict mapping word to count
    word_counts[word] = word_counts.get(word, 0) + 1

word_counts = sorted(list(word_counts.items()), key=lambda i: (-i[1], i[0])) #convert dict to list of tuples sort by count then word

less_than_min = 0
for i in range(len(word_counts) - 1, -1, -1):
    if word_counts[i][1] < MIN_COUNT:
        less_than_min += word_counts[i][1]
        del word_counts[i]

word_counts.append(("<UNK>", less_than_min))
word_counts.sort(key=lambda i: (-i[1], i[0])) #resort for <UNK>

#https://towardsdatascience.com/generating-text-with-tensorflow-2-0-6a65c7bdc568

vocab = [i[0] for i in word_counts] #list of all words
words = [w if w in vocab else "<UNK>" for w in words] #sets words not in vocab to <UNK>

word2int = {w:i for i, w in enumerate(vocab)}
int2word = np.array(vocab)
#print("Vector:\n")
#for word,_ in zip(word2int, range(len(vocab))):
#    print(" {:4s}: {:3d},".format(repr(word), word2int[word]))

words_as_ints = np.array([word2int[w] for w in words], dtype=np.int32)
#print("{}\n mapped to integers:\n {}".format(repr(words[:100]), words_as_ints[:100]))

batch_size = 64
buffer_size = 10000
embedding_dim = 256
epochs = 50
seq_length = 200
examples_per_epoch = len(words) // seq_length
#lr = 0.001 #will use default for Adam optimizer
rnn_units = 1024

train_size = 0
while (train_size <= 0.9 * len(words_as_ints) - batch_size):
    train_size += batch_size

train_words = words_as_ints[:train_size]
test_words = words_as_ints[train_size:]
#print(words_as_ints.shape, train_words.shape, test_words.shape)

train_word_dataset = tf.data.Dataset.from_tensor_slices(train_words)
test_word_dataset = tf.data.Dataset.from_tensor_slices(test_words)

train_sequences = train_word_dataset.batch(seq_length + 1, drop_remainder=True)
test_sequences = test_word_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_words = chunk[:-1]
    target_words = chunk[1:]
    return input_words, target_words

train_dataset = train_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
test_dataset = test_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)

#print("\n\n", train_dataset, test_dataset)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(vocab))
])

print("\n\n\n")
model.summary()