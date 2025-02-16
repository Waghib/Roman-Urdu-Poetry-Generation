import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras  

# Define custom perplexity metric
def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.exp(tf.reduce_mean(cross_entropy))

# Load trained model
@st.cache_resource
def load_model():
    return keras.models.load_model("roman_urdu_poetry_model.keras", custom_objects={"perplexity": perplexity})

model = load_model()

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function to generate poetry
def generate_text(seed_word, next_words=10, max_seq_len=50, temperature=0.8):
    def generate_verse(seed_text, next_words):
        for _ in range(next_words):
            # Tokenize and pad sequence
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')

            # Predict next word probabilities
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted_probs = np.log(predicted_probs + 1e-8) / temperature
            predicted_probs = np.exp(predicted_probs - np.max(predicted_probs))
            predicted_probs /= np.sum(predicted_probs)

            # Sample a word based on probabilities
            predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
            output_word = tokenizer.index_word.get(predicted_index, "")

            if output_word:
                seed_text += " " + output_word

        return seed_text

    # Generate first verse
    verse1 = generate_verse(seed_word, next_words)

    # Use last 3 words of verse1 as seed for the second verse
    last_words = " ".join(verse1.split()[-3:])
    verse2 = generate_verse(last_words, next_words)

    return verse1, verse2

# ===============================
# 📌 Streamlit UI
# ===============================

st.title("📜 Urdu Poetry Generator 🎤")
st.write("Enter a seed word or phrase, and AI will generate poetic verses.")

# User input
seed_word = st.text_input("Enter a starting word:", "Muhabbat")

# Slider for output length selection
next_words = st.slider("Select number of words to generate:", min_value=5, max_value=20, value=10, step=1)

if st.button("✨ Generate Poetry"):
    if seed_word.strip():
        verse1, verse2 = generate_text(seed_word, next_words)
        
        # Display poetry
        st.subheader("📌 Generated Poetry:")
        st.write(f"📖 *{verse1}*")
        st.write(f"📖 *{verse2}*")

        # Text-to-Speech using JavaScript inside Streamlit
        js_code = f"""
        <script>
        function speakPoetry() {{
            var msg = new SpeechSynthesisUtterance();
            msg.text = `{verse1} {verse2}`;
            msg.lang = "ur-PK";  // Urdu language
            msg.rate = 0.9;  // Speed
            window.speechSynthesis.speak(msg);
        }}
        </script>
        <button onclick="speakPoetry()" style="background-color:#4CAF50;color:white;padding:10px;border:none;border-radius:5px;cursor:pointer;">
        🔊 Listen to Poetry
        </button>
        """
        
        # Inject JavaScript using Streamlit Components
        st.components.v1.html(js_code, height=50)

    else:
        st.warning("⚠️ Please enter a valid word to generate poetry!")
