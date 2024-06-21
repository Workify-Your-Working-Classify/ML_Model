import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pickle


file_path = 'data - Sheet1 (2).csv'
data = pd.read_csv(file_path)

# Extract features and labels
sentences = data['Name'].astype(str).tolist()
labels = data['Label'].astype(int).tolist()

# Split data into training and testing sets (80:20 ratio)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42)

# Parameters
vocab_size = 5000
embedding_dim = 64
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Tokenizing and padding sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

tokenizer_file = "tokenizer.pickle"
with open(tokenizer_file, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Define the model with adjustments
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
    Dropout(0.5),
    Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), metrics=['accuracy'])

# Summary of the model
model.summary()

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Early stopping callback with increased patience
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(train_padded, train_labels, epochs=50, verbose=2, validation_data=(test_padded, test_labels), callbacks=[early_stopping])

model.save("Model.h5")

# Plotting the training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_padded, test_labels, verbose=2)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Hypothetical test data for predictions
sample_sentences = [
    'Prepare quarterly financial reports for the board meeting',
    'Finalize the marketing strategy for the upcoming product launch',
    'Conduct performance evaluations for team members',
    'Review and update the company cybersecurity protocols',
    'Attend a crucial client meeting to discuss project milestones and deliverables',
    'Organize your desktop icons',
    'Rearrange the furniture in your office',
    'Clean out your email inbox',
    'Water the plants in the office',
    'Update your social media profile picture'
]
sample_sequences = tokenizer.texts_to_sequences(sample_sentences)
sample_padded = pad_sequences(sample_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Predictions
sample_predictions = model.predict(sample_padded)

for i, prediction in enumerate(sample_predictions):
    print(f"Sentence: '{sample_sentences[i]}' - Predicted Label: {int(round(prediction[0]))}")
