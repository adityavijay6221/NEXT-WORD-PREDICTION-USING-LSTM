# NEXT-WORD-PREDICTION-USING-LSTM
## 1. Problem Statement
Predicting the next word in a given text sequence is a fundamental task in Natural Language Processing (NLP) and is the foundation for applications such as autocomplete, text generation, and conversational AI.

This project focuses on building a Next Word Prediction model using Long Short-Term Memory (LSTM) networks trained on William Shakespeare’s Hamlet, aiming to capture complex language dependencies and poetic structure.

The goal is to take an input sequence of words (e.g., “To be or not”) and accurately predict the next most probable word (e.g., “to”).
## 2. EDA and Preprocessing
Dataset
The model uses Hamlet by William Shakespeare as the training corpus.
The text is stored in hamlet.txt, containing raw dialogue and prose from the play.
Steps in Data Preparation
## Text Cleaning:
- Lowercased the entire text.
- Removed punctuation, numbers, and special symbols.
- Tokenized text into words using NLTK.
## Sequence Creation:
- Created sequences of words where each sequence predicts the next word.
Example:
Input: ['to', 'be', 'or', 'not']
Output: 'to'
## Encoding:
- Used Keras Tokenizer to convert words into integer tokens.
- Saved tokenizer as tokenizer.pkl for later use.
## Padding:
- Applied pad_sequences() to ensure uniform sequence length before feeding into the model.
## Training Data:
- X → Sequences of tokens.
- y → One-hot encoded target words.

## 3. Model Architecture and Training
The model is based on LSTM, a type of Recurrent Neural Network (RNN) that captures long-term dependencies in text.
| Layer               | Type                                                      | 
| ------------------- | --------------------------------------------------------- |
| **Embedding**       | Maps word indices to dense vectors of fixed size          |  
| **LSTM**            | Learns contextual dependencies in sequences               |       
| **Dense (Softmax)** | Outputs probability distribution for next word prediction |   

### Training Configuration
- Loss Function: categorical_crossentropy
- Optimizer: adam
- Metric: accuracy
- Epochs: 50 (until convergence)
- Batch Size: 128
The trained model is saved as model_lstm.h5.

## 4. Evaluation Process and Insights
Evaluation Metrics
- Training Accuracy: ~92%
- Validation Accuracy: ~89%
- Loss Curve: Gradual decrease indicating stable learning.
  
### Key Observations
- The LSTM successfully captures contextual flow in poetic text.
- Performance improves with larger datasets (more plays or corpus).
- Rare Shakespearean words are harder to predict due to low frequency.

## 5. Deployment and Impact
### Deployment
The model is deployed using Streamlit (app.py) for interactive inference.

### App Workflow
- User enters a partial sentence (e.g., “The lady doth protest”).
- The text is tokenized and padded using the saved tokenizer.pkl.
- The trained model_lstm.h5 predicts the most probable next word.
- The app displays the predicted word dynamically.

### Impact
- Demonstrates how deep learning can model literary writing styles.
- Can be extended to creative text generation or chatbot applications.
- Serves as a foundation for language modeling and sequence prediction tasks.

## Technologies Used
- Python
- TensorFlow/Keras
- Streamlit
- NLTK
- NumPy & Pandas


