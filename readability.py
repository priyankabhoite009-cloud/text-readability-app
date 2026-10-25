import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
import os

st.set_page_config(page_title="Text Readability Predictor", layout="centered")

st.title("📖 AI-Powered Text Readability Predictor")
st.write("Enter an excerpt below to evaluate its complexity score using an LSTM Neural Network.")

# Automatically build, train, or load the model
@st.cache_resource
def get_model_and_tokenizer():
    model_path = "readability_model.keras"
    
    sample_texts = [
        "The solar system consists of the sun and the planets orbiting it.",
        "Quantum mechanics explores the wave-particle duality at subatomic scales.",
        "Plants use sunlight to convert carbon dioxide and water into glucose.",
        "Macroeconomic indicators such as GDP and inflation measure national economic health.",
        "The cat sat lazily on the warm wooden porch.",
        "Epistemological frameworks dictate how empirical knowledge is validated.",
        "Water boils at one hundred degrees Celsius under standard atmospheric pressure.",
        "Neurological pathways transmit electrical impulses throughout the central nervous system."
    ]
    sample_targets = np.array([1.2, -2.1, 0.8, -1.8, 1.5, -2.4, 1.1, -2.0])
    
    vocab_size = 1000
    max_length = 20
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(sample_texts)
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        sequences = tokenizer.texts_to_sequences(sample_texts)
        padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(padded, sample_targets, epochs=25, verbose=0)
        model.save(model_path)
        
    return model, tokenizer, max_length

model, tokenizer, max_length = get_model_and_tokenizer()

# User Input Form
user_input = st.text_area("Enter your text passage here:")

if st.button("Evaluate Readability"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
        score = model.predict(padded)[0][0]
        
        st.subheader("Results:")
        st.metric(label="Readability Score", value=round(float(score), 3))
        
        if score > 0:
            st.success("🟢 **Level: Easy / Beginner Friendly**")
        else:
            st.error("🔴 **Level: Advanced / Academic**")