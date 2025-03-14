import os
import string
import numpy as np
import pandas as pd
import torch
import spacy
import nltk
from collections import Counter
# from textwrap import wrapimport
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
# NLTK Modules
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag

# Ensure necessary NLTK datasets are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')  # Download missing 'punkt_tab' package

# Load SpaCy model

import streamlit as st

# Load the model from the local path
model_path = os.path.join(os.getcwd(), "en_core_web_sm")
try:
    nlp = spacy.load(model_path)
    print("SpaCy model loaded successfully!")
except Exception as e:
    print(f"Failed to load SpaCy model: {e}")



nlp = spacy.load("en_core_web_sm")
st.title('AI Generated Detection')
text_input = st.text_area('Enter text',height=300)

#creating linguistic features dataframe from given text input
def calculate(text):
    def tokenize_text(text):
        if not isinstance(text, str):
            return []
        return word_tokenize(text)

    def tokenize_sent(text):
        if not isinstance(text, str):
            return []
        sentences = sent_tokenize(text)
        return [len(sent.split()) for sent in sentences]

    def sent_wala(text):
        if not isinstance(text, str):
            return []
        return sent_tokenize(text)

    def vocab_for_row(words):
        return len(set(words))

    def number_of_sentence(sentences):
        return len(sentences)

    def number_of_words(words):
        return len(words)

    def count_stop_words(words):
        doc = nlp(" ".join(words))
        return sum(1 for token in doc if token.is_stop)

    def is_passive(sentence):
        for token in sentence:
            if token.dep_ == "nsubjpass":
                return True
        return False

    def count_active(sentences):
        active_count = 0
        passive_count = 0
        for sent in sentences:
            doc = nlp(sent)
            if is_passive(doc):
                passive_count += 1
        active_count = len(sentences) - passive_count
        return active_count

    def count_nouns_verbs(word_list):
        tags = pos_tag(word_list)
        counts = Counter(tag for word, tag in tags)
        pos_counts = {
            'NOUN': counts.get('NN', 0) + counts.get('NNS', 0) + counts.get('NNP', 0) + counts.get('NNPS', 0),
            'VERB': counts.get('VB', 0) + counts.get('VBD', 0) + counts.get('VBG', 0) + counts.get('VBN', 0) + counts.get('VBP', 0) + counts.get('VBZ', 0),
            'PUNCT': counts.get('.', 0) + counts.get(',', 0) + counts.get(':', 0) + counts.get('(', 0) + counts.get(')', 0) + counts.get('"', 0) + counts.get("''", 0) + counts.get('``', 0) + counts.get('!', 0) + counts.get('?', 0) + counts.get(';', 0) + counts.get('-', 0),
            'DET': counts.get('DT', 0) + counts.get('PDT', 0) + counts.get('WDT', 0),
            'PRON': counts.get('PRP', 0) + counts.get('PRP$', 0),
            'PROPN': counts.get('NNP', 0) + counts.get('NNPS', 0),
            'ADJ': counts.get('JJ', 0) + counts.get('JJR', 0) + counts.get('JJS', 0),
            'AUX': counts.get('MD', 0),
            'ADV': counts.get('RB', 0) + counts.get('RBR', 0) + counts.get('RBS', 0),
            'PART': counts.get('RP', 0),
            'SCONJ': counts.get('IN', 0),
            'NUM': counts.get('CD', 0),
            'X': counts.get('FW', 0),
            'INTJ': counts.get('UH', 0),
            'ADP': counts.get('IN', 0),
            'SYM': counts.get('SYM', 0),
            'SPACE': counts.get('SP', 0),
            'CCONJ': counts.get('CC', 0)
        }
        return pos_counts

    def count_punctuation_marks(text):
        return sum(1 for char in text if char in string.punctuation)

    def count_linking_words(text):
        linking_words = {'to', 'the', 'and', 'of', 'in', 'on', 'for', 'with', 'at', 'a', 'an'}
        return sum(1 for word in word_tokenize(text.lower()) if word in linking_words)

    words = tokenize_text(text)
    sentences = sent_wala(text)
    sentence_lengths = tokenize_sent(text)
    vocab = vocab_for_row(words)
    line_count = number_of_sentence(sentences)
    word_count = number_of_words(words)
    avg_line_length = word_count / line_count if line_count > 0 else 0
    stopwords_count = count_stop_words(words)
    stopwords_per = (stopwords_count / word_count * 100) if word_count > 0 else 0
    active_count = count_active(sentences)
    passive_count = line_count - active_count
    active_per = (active_count / line_count * 100) if line_count > 0 else 0
    passive_per = (passive_count / line_count * 100) if line_count > 0 else 0
    pos_counts = count_nouns_verbs(words)
    punctuation_count = count_punctuation_marks(text)
    punc_per = (punctuation_count / word_count * 100) if word_count > 0 else 0
    linking_words_count = count_linking_words(text)
    link_per = (linking_words_count / word_count * 100) if word_count > 0 else 0
    word_density = (vocab * 100) / (avg_line_length * line_count) if avg_line_length * line_count > 0 else 0
    return words, sentences, sentence_lengths, vocab, line_count, word_count, avg_line_length, stopwords_count, stopwords_per, active_count, passive_count, active_per, passive_per, pos_counts,pos_counts, punctuation_count, linking_words_count, punc_per, link_per, word_density

def analyze_text(text):
    words, sentences, sentence_lengths, vocab, line_count, word_count, avg_line_length, stopwords_count, stopwords_per, active_count, passive_count, active_per, passive_per, pos_counts,pos_counts, punctuation_count, linking_words_count, punc_per, link_per, word_density = calculate(text)

    data = {
        "Words": [words],
        "Sentences": [sentences],
        "Sentence Lengths": [sentence_lengths],
        "Vocab Count": [vocab],
        "Line Count": [line_count],
        "Word Count": [word_count],
        "Avg Line Length": [avg_line_length],
        "Stopwords Count": [stopwords_count],
        "Stopwords Percentage": [stopwords_per],
        "Active Sentences": [active_count],
        "Passive Sentences": [passive_count],
        "Active Percentage": [active_per],
        "Passive Percentage": [passive_per],
        "Linking Words Count": [linking_words_count],
        "Punctuation Count": [punctuation_count],
        "Punctuation Percentage": [punc_per],
        "Linking Words Percentage": [link_per],
        "Word Density": [word_density]
    }

    # Add all POS counts dynamically
    for key, value in pos_counts.items():
        data[key] = [value]

    df = pd.DataFrame(data)
    return df
kdf = analyze_text(text_input)



# Define the column mapping from df1 to df2 (if names differ)
column_rename_map = {
    'Vocab Count': 'vocab',
    'Line Count': 'line count',
    'Word Count': 'word count',
    'Avg Line Length': 'avg line length',
    'Stopwords Count': 'stopwords_count',
    'Active Sentences': 'active',
    'Passive Sentences': 'passive',
    'Punctuation Count': 'punctuation_count',
    'Linking Words Count': 'linking_words_count',
    'Punctuation Percentage': 'punc_per',
    'Linking Words Percentage': 'link_per',
    'Word Density': 'word density'
}




#column order
# Rename the columns in df1
df1_renamed = kdf.rename(columns=column_rename_map)

# Ensure only the columns from df2 are selected
df2_columns = ['avg line length', 'vocab', 'word density', 'stopwords_count', 'word count',
               'active', 'passive', 'punctuation_count', 'linking_words_count', 'NOUN', 'VERB',
               'PUNCT', 'DET', 'PRON', 'PROPN', 'ADJ', 'AUX', 'ADV', 'PART', 'SCONJ', 'NUM',
               'X', 'INTJ', 'ADP', 'SYM', 'SPACE', 'CCONJ']

# Keep only columns present in df2 and reorder
final_df = df1_renamed[df1_renamed.columns.intersection(df2_columns)]
final_df = final_df.reindex(columns=df2_columns, fill_value=0)  # Ensure order & fill missing column

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Define classification model
class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # No Sigmoid here; use BCEWithLogitsLoss
        return x

# Define input size (768 embedding + 27 numerical features)
input_size = 795
model = ClassificationModel(input_size).to(device)

# Load trained model weights
model.load_state_dict(torch.load("classification_model1.pth", map_location=device))
model.eval()

# Load trained scaler (saved during training)
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy")  # Load mean
scaler.scale_ = np.load("scaler_scale.npy")  # Load scale

# Load Sentence Transformer model
sentence_model = SentenceTransformer("all-mpnet-base-v2").to(device)

# List of required numerical features (must match training order)
FEATURE_COLUMNS = [
    "avg line length", "vocab", "word density", "stopwords_count",
    "word count", "active", "passive", "punctuation_count",
    "linking_words_count", "NOUN", "VERB", "PUNCT", "DET", "PRON", "PROPN",
    "ADJ", "AUX", "ADV", "PART", "SCONJ", "NUM", "X", "INTJ", "ADP", "SYM",
    "SPACE", "CCONJ"
]

if "text_input_key" not in st.session_state:
    st.session_state.text_input_key = ""


if text_input:
    # Function for inference
    
    def clear_text():
        st.session_state.text_input_key = ""
    col1, col2 = st.columns(2)
    with col1:
        # if st.button("Submit"):
            # st.write(f"You entered: {text_input}")

        if st.button("Submit"):

            def predict(text_list, numerical_df):
                """
                text_list: List of texts (sentences/documents)
                numerical_df: DataFrame with 27 numerical features (same order as training)
                Returns:
                    predictions: Binary classification labels (0 or 1)
                    probabilities: Predicted probability scores
                """
                # Ensure numerical_df has the correct feature order
                numerical_df = numerical_df[FEATURE_COLUMNS]

                # Generate text embeddings
                text_embeddings = sentence_model.encode(text_list, convert_to_tensor=True).to(device)
                # numerical_features = numerical_features.reshape(text_embeddings.shape[0], -1) # Reshape


                # Standardize numerical features
                numerical_features = torch.tensor(scaler.transform(numerical_df), dtype=torch.float32).to(device)
                # numerical_features = numerical_features.reshape(text_embeddings.shape[0], -1)
                numerical_features = numerical_features.reshape(len(text_list), -1)


                # Concatenate text embeddings (768) and numerical features (27) -> 795-dimensional input
                X_combined = torch.cat((text_embeddings, numerical_features), dim=1)

                # Make predictions
                with torch.no_grad():
                    outputs = model(X_combined)
                    probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()

                # Convert probabilities to binary predictions
                predictions = (probabilities > 0.5).astype(int)

                return predictions, probabilities

            predictions, probs = predict([text_input], final_df)
            # Simulated probability (replace with model output)
            # Example: Probability that text is AI-generated


            # st.write(f"**Probability of AI-generated text: {float(probs):.2%}**")

            st.write(f"**Probability of AI-generated text: {float(probs[0]):.2%}**")  # Extract first element if it's an array

            st.progress(float(probs[0]))  # Show probability visually
    
    with col2:
        st.button("Clear", on_click=clear_text)

else:
    st.write('Please enter input')
