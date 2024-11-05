# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import random

app = Flask(__name__)

# Load the pre-trained model and character mappings
model = tf.keras.models.load_model('shakespeare_generation_v3.keras')
with open("shakespeare_text.txt", "r") as file:
    text = file.read().lower()

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}
sequence_length = 80

# Sampling function
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Text generation function
def generate_text(input_text, length=300, temperature=0.8):
    generated = input_text
    for _ in range(length):
        x_pred = np.zeros((1, sequence_length), dtype=np.int32)
        for t, char in enumerate(generated[-sequence_length:]):
            x_pred[0, t] = char_to_index.get(char, 0)

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char

    return generated

# Define a root endpoint to serve a simple HTML page
@app.route('/')
def home():
    return '''
        <h1>Shakespeare Text Generator</h1>
        <p>Use the <a href="/generate">/generate</a> endpoint to generate text.</p>
    '''

# Define an endpoint for text generation
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('input_text', '')
    temperature = float(data.get('temperature', 0.8))
    length = int(data.get('length', 300))

    generated_text = generate_text(input_text, length=length, temperature=temperature)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
