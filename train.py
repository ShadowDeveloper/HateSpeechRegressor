import time

print("Loading libraries...")
start_time = time.time()

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import datasets
import pickle

print(f"Libraries loaded in {round((time.time() - start_time) * 1000, 3)} ms.")
print("Loading vectorizer...")
start_time = time.time()

count_vect = CountVectorizer()

print(f"Vectorizer loaded in {round((time.time() - start_time) * 1000, 3)} ms.")
print(f"Saving vectorizer...")
start_time = time.time()

# Save vectorizer
pickle.dump(count_vect, open('vectorizer.pkl', 'wb'))


print("Setting configuration...")
start_time = time.time()

# Set configuration
sklearn.set_config(working_memory=4096)
data_size = 100000


print(f"Configuration set in {round((time.time() - start_time) * 1000, 3)} ms.")
print("Loading data...")
start_time = time.time()

# Load data
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
df = dataset['train'].to_pandas()

print(f"Data loaded in {round((time.time() - start_time) * 1000, 3)} ms.")
print(df.head())

print("Preprocessing data...")
start_time = time.time()

# Extract text and labels
X_text = df['text'][:data_size]  # Assuming 'text' is the column containing the text data
y_columns = ['hate_speech_score', 'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech']
y = df[y_columns][:data_size]
y = y.fillna(0)

# Convert text to vectors
X = count_vect.fit_transform(X_text)

print(f"Data preprocessed in {round((time.time() - start_time) * 1000, 3)} ms.")
print("Splitting data...")
start_time = time.time()
# Load data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(f"Data split in {round((time.time() - start_time) * 1000, 3)} ms.")
print("Training model...")
start_time = time.time()

# Create MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32, 16), activation='relu', max_iter=100, alpha=0.0001, learning_rate_init=0.003, solver='adam', verbose=True, tol=0.000000000001, early_stopping=False, n_iter_no_change=5000)
mlp.fit(X_train, y_train)

print(f"Model trained in {round((time.time() - start_time), 3)} s.")
print("Evaluating model...")

# Predict and score
predictions = mlp.predict(X_test)
print("Mean squared error: ", mean_squared_error(y_test, predictions))

# Plot the loss curve
plt.plot(mlp.loss_curve_)
plt.title("Loss curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

print("Done!")

# Save the model to disk

filename = 'model.pkl'
pickle.dump(mlp, open(filename, 'wb'))

# Test the model for fun :)
sentences = count_vect.fit_transform(["Fuck you you stupid nigger", "You're a piece of shit", "Awesome!", "Oh my god, I never realized that!"])

predictions = mlp.predict(sentences)
# Write dict of sentences and predictions
values = {sentences[i]: predictions[i] for i in range(len(sentences))}

