import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation

# Load dataset
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Use a subset of the text
text = text[300000:800000]

# Character mapping
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 3

# Load the model
model = tf.keras.models.load_model('textgenerator.model')

# Sampling function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')  # Fixed typo
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Text generation function
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))  # Fixed shape issue
        for t, char in enumerate(sentence):  # Renamed loop variable
            x[0, t, char_to_index[char]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]  # Fixed typo

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

# Generate text with different temperatures
print('----------0.2----------')
print(generate_text(300, 0.2))
print('----------0.4----------')
print(generate_text(300, 0.4))
print('----------0.6----------')
print(generate_text(300, 0.6))
print('----------0.8----------')
print(generate_text(300, 0.8))
print('----------1----------')
print(generate_text(300, 1))
