import nltk
import tensorflow 
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#LOADING THE MODEL
model=load_model('model_lstm.h5')

#LOADING THE TOKENIZER
with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)
    
#Function to make predictions
def predict_word(model,tokenizer,text_seq,max_len):
  token_list=tokenizer.texts_to_sequences([text_seq])[0]
  token_list=pad_sequences([token_list],maxlen=max_len-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(predicted,axis=1) #predicted_word_index=np.argmax(predicted,axis=1): The output of the model is a probability distribution over all possible words in the vocabulary. np.argmax finds the index of the word with the highest predicted probability.
  for word,index in tokenizer.word_index.items():
    if index==predicted_word_index:
      return word
  return None

#STREAMLIT APP
st.title('Next Word Prediction with LSTM')

input_text=st.text_input('Enter the sequence of words')
if st.button('Predict next word'):
    max_seq_len=model.input_shape[1]+1
    predicted_word=predict_word(model,tokenizer,input_text,max_seq_len)
    st.write(f'Next word: {predicted_word}')