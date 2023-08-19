import time

print("Loading libraries...")
start_time = time.time()

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import datasets
import pickle

print(f"Libraries loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Setting configuration...")
start_time = time.time()

# Set configuration
sklearn.set_config(working_memory=4096)
data_size = 100000


print(f"Configuration set in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Loading model and vectorizer...")
start_time = time.time()

with open('model.pkl', 'rb') as model_file:
    mlp = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    count_vect = pickle.load(vectorizer_file)

print(f"Model and vectorizer loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Loading data...")
start_time = time.time()

# Load data
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
df = dataset['train'].to_pandas()

print(f"Data loaded in {round((time.time() - start_time) * 1000, 3)} ms.")
print(df.head())

print("Fitting vectorizer...")
start_time = time.time()

# Extract text and labels
X_text = df['text'][:data_size]  # Assuming 'text' is the column containing the text data

# Convert text to vectors
X = count_vect.fit(X_text)

print(f"Vectorizer fit in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Predicting...")
start_time = time.time()

sentences = ["Fuck you you stupid nigger", "You're a piece of shit", "Awesome!", "Oh my god, I never realized that!"]
vectorized_sentences = count_vect.transform(sentences)

predictions = mlp.predict(vectorized_sentences)
# write dict of sentences and predictions
values = {sentences[i]: predictions[i] for i in range(len(sentences))}
print(values)
