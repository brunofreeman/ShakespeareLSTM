import tensorflow as tf #2.0.0
import numpy as np
import json
import os
import re
import datetime

#Set Root and Version
ROOT = "."
LSTM_VERSION = 1
settings = json.load(open(os.path.join(ROOT, "version_settings", "v" + str(LSTM_VERSION) + "_settings.json")))

#Data Settings
USING_WORDS = settings["USING_WORDS"]
MIN_UNIT_COUNT = settings["MIN_UNIT_COUNT"]

#Training Settings
BATCH_SIZE = settings["BATCH_SIZE"]
BUFFER_SIZE = settings["BUFFER_SIZE"]
EMBEDDING_DIM = settings["EMBEDDING_DIM"]
EPOCHS = settings["EPOCHS"]
SEQ_LEN = settings["SEQ_LEN"]
RNN_UNITS = settings["RNN_UNITS"]
PATIENCE = settings["PATIENCE"]
TRAIN_PERCENT = settings["TRAIN_PERCENT"]

#File Settings
DATA_DIR = os.path.join(ROOT, settings["DATA_DIR"])
CKPT_DIR = os.path.join(ROOT, "checkpoinnts", settings["CKPT_DIR"])
OUTPUT_DIR = os.path.join(ROOT, "lstm_output", settings["OUTPUT_DIR"])

def get_time_for_file():
    return datetime.datetime.now().strftime("_%m.%d.%y-%H.%M.%S")

def get_ckpt_prefix():
    return os.path.join(CKPT_DIR, "ckpt" + get_time_for_file())

#Generation Settings
SEED = "a great tale"
PRINT_TO_FILE = True
NUM_UNITS_GENERATE = 1000
TEMPERATURE = 1.0

text = ""
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        text += open(os.path.join(DATA_DIR, file)).read().lower()

regex = r"(?:[A-Za-z']*(?:(?<!-)-(?!-))*[A-Za-z']+)+" + r"|\.|\?|!|,|;|:|-|\(|\)|\[|\]|\{|\}|\'|\"|\|\/|<|>| |\t|\n" if USING_WORDS else r".|\n"
units = re.findall(regex, text)
unit_counts = dict()

for unit in units: #create a dict mapping unit to count
    unit_counts[unit] = unit_counts.get(unit, 0) + 1

unit_counts = sorted(list(unit_counts.items()), key=lambda i: (-i[1], i[0])) #convert dict to list of tuples sort by count then unit

less_than_min = 0
for i in range(len(unit_counts) - 1, -1, -1):
    if unit_counts[i][1] < MIN_UNIT_COUNT:
        less_than_min += unit_counts[i][1]
        del unit_counts[i]

unit_counts.append(("<UNK>", less_than_min))
unit_counts.sort(key=lambda i: (-i[1], i[0])) #resort for <UNK>

vocab = [i[0] for i in unit_counts] #list of all units
units = [w if w in vocab else "<UNK>" for w in units] #sets units not in vocab to <UNK>

unit2int = {w:i for i, w in enumerate(vocab)}
int2unit = np.array(vocab)

units_as_ints = np.array([unit2int[w] for w in units], dtype=np.int32)

def split_input_target(chunk):
    input_units = chunk[:-1]
    target_units = chunk[1:]
    return input_units, target_units

def build_model(embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(len(vocab), embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(vocab))
    ])

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train_model():
    examples_per_epoch = len(units) // SEQ_LEN

    train_size = 0
    while (train_size <= TRAIN_PERCENT * len(units_as_ints) - BATCH_SIZE):
        train_size += BATCH_SIZE

    train_units = units_as_ints[:train_size]
    test_units = units_as_ints[train_size:]

    train_unit_dataset = tf.data.Dataset.from_tensor_slices(train_units)
    test_unit_dataset = tf.data.Dataset.from_tensor_slices(test_units)

    train_sequences = train_unit_dataset.batch(SEQ_LEN + 1, drop_remainder=True)
    test_sequences = test_unit_dataset.batch(SEQ_LEN + 1, drop_remainder=True)

    train_dataset = train_sequences.map(split_input_target).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    test_dataset = test_sequences.map(split_input_target).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    model = build_model(EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    model.load_weights(tf.train.latest_checkpoint(CKPT_DIR))

    print("Loaded checkpoint " + tf.train.latest_checkpoint(CKPT_DIR) + "\n")

    model.summary()

    for input_example_batch, target_example_batch in train_dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape)

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Loss: ", example_batch_loss.numpy().mean())

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=get_ckpt_prefix(), save_weights_only=True)

    history = model.fit(train_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback, early_stop], validation_data=test_dataset)

    print("Training stopped due to no improvement after %d epochs" % PATIENCE)

def generate_text(model, seed):
    seed = re.findall(regex, seed)
    input_eval = [unit2int[s] for s in seed]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()

    for i in range(NUM_UNITS_GENERATE):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / TEMPERATURE
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(int2unit[predicted_id])
    return "".join(text_generated)

def run_model(seed):
    model = build_model(EMBEDDING_DIM, RNN_UNITS, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(CKPT_DIR))
    model.build(tf.TensorShape([1, None]))

    print("Generating with Shakespeare LSTM v" + str(LSTM_VERSION))
    print("Checkpoint: " + tf.train.latest_checkpoint(CKPT_DIR))
    print("Seed: " + seed)

    output = seed + generate_text(model, seed)

    if PRINT_TO_FILE:
        file_name = os.path.join(OUTPUT_DIR, "output" + get_time_for_file() + ".txt")
        with open(file_name, "w") as output_file:
            output_file.write(output)
        print("Generated text saved to " + file_name)
    else:
        print("\n")
        print(output)

#run_model(SEED)