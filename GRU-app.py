from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import pickle

with open('GRU_tokenizer.pkl','rb') as handle:
    tokenizer=pickle.load(handle)

model=load_model('GRU_100.h5')

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next word Prediction")

input= st.text_input("Enter Text Here", value="")

if st.button('Predict Next word'):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input,max_sequence_len)
    st.write(f"Next word:{next_word}")
    