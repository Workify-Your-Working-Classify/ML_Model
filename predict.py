import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import os

# Set environment variables for encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'Model.h5')
custom_objects = {
    'InputLayer': tf.keras.layers.InputLayer  # Ensure this matches your model's configuration
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Parameters (ensure these are the same as used during training)
vocab_size = 5000
max_length = 120
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"

# Load the tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.pickle')
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to classify a single sentence
def classify_sentence(sentence):
    # Tokenize and pad the input sentence
    input_sequence = tokenizer.texts_to_sequences([sentence])
    input_padded = pad_sequences(input_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Predictions
    prediction = model.predict(input_padded)
    predicted_label = int(round(prediction[0][0]))

    # Map labels to meaningful strings
    if predicted_label == 1:
        return "Important"
    else:
        return "Not Important"

# Main function to handle JSON input/output
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python classify_sentence.py <input_json_file> <output_json_file>")
        sys.exit(1)

    input_json_file = sys.argv[1]
    output_json_file = sys.argv[2]

    # Read the input JSON from file with UTF-8 encoding
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read input file: {str(e)}"}))
        sys.exit(1)

    # Check if 'sentences' key is present in the input JSON
    if 'sentences' not in data:
        print(json.dumps({"error": "Input JSON must contain 'sentences' key"}))
        sys.exit(1)

    input_sentences = data['sentences']
    output_predictions = []

    # Classify each input sentence
    for sentence in input_sentences:
        try:
            # Ensure the sentence is encoded properly
            sentence = sentence.encode('utf-8').decode('utf-8')
            predicted_label = classify_sentence(sentence)
            output_predictions.append({
                "sentence": sentence,
                "predicted_label": predicted_label
            })
        except Exception as e:
            output_predictions.append({
                "sentence": sentence,
                "error": str(e)
            })

    # Prepare the output JSON
    output = {
        "predictions": output_predictions
    }

    # Write the output JSON to file with UTF-8 encoding
    try:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False)
    except Exception as e:
        print(json.dumps({"error": f"Failed to write output file: {str(e)}"}))
        sys.exit(1)

    print(f"Output written to {output_json_file}")

#Prediction python code, used in cmd