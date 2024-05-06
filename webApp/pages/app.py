import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
# from report import report_page
# from test_model import test_model_page
# from search import search_page
import pandas as pd
import matplotlib.pyplot as plt
#from langdetect import detect
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from wordcloud import WordCloud
#from collections import Counter
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import joblib
from transformers import pipeline, RobertaTokenizer
import pickle
from pathlib import Path
from zipfile import ZipFile 

rootPath = str(Path(__file__).resolve().parent.parent)
print(rootPath)

def unzipFile(path):
    with ZipFile(path, 'r') as zObject: 
        zObject.extractall(rootPath)
    zObject.close()

def clean_text(text):
    # Clean text data in lyrics column
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def clean_text_sentiment(text,negation_words):
    # Clean text data in lyrics column
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.difference(negation_words)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def load_glove_embeddings(embedding_path):
    """
    Load GloVe embeddings from a file.

    Args:
    - embedding_path (str): Path to the GloVe embedding file.

    Returns:
    - word_embeddings (dict): Dictionary mapping words to their GloVe vectors.
    """
    word_embeddings = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = vector
    return word_embeddings

# Function to get embedding for each word
def get_word_embedding(word,word_embeddings):
    return word_embeddings.get(word, np.zeros_like(next(iter(word_embeddings.values()))))  # Use the size of any vector from word_embeddings

# Function to get average embedding for a lyric
def get_lyric_embedding(lyric,word_embeddings):
    words = lyric.split()  # Tokenize lyric into words
    embeddings = [get_word_embedding(word,word_embeddings) for word in words]  # Get embeddings for each word
    if embeddings:
        return np.mean(embeddings, axis=0)  # Return average embedding of all words
    else:
        return np.zeros_like(next(iter(word_embeddings.values())))  # Return zero vector if lyric is empty

def getTopSentiments(text):
    """
    Get top sentiments from the text using RoBERTa-based sentiment classifier.
    """
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)
    
    while len(tokenizer.tokenize(text)) > 510:
        text = text[:len(text) - 10]
    sentiments_list = classifier(text)[0]
    x = sentiments_list[0]['label']
    if sentiments_list[1]['score'] > 0.2:
        x = x + "," + sentiments_list[1]['label']
    if sentiments_list[2]['score'] > 0.2:
        x = x + "," + sentiments_list[2]['label']
    #print[x]
    return x


def report_page():
    # Main page title
    st.title("Report Page")

    # Side heading using Markdown syntax
    # st.markdown("## Introduction")
    st.markdown("<h2 style='font-size: 30px;'>Introduction</h2>", unsafe_allow_html=True)

    # Main page content
    st.write("Our project's goal is to use song lyrics to categorize music into genres. Humans find it difficult to accomplish this task, and since borders are not always obvious, there is frequent discussion regarding where song fits in. Music genres show similarities between tracks, which helps to group music into collections. Songs often fit into more than one genre, indicating that genre isn't always clearly defined. Automating this classification process is highly motivated by technologies such as Spotify, which adds an estimated 60,000 songs to its database every day.")

     # Preprocessing details
    st.write("""
    We have merged the data, dropped all non-English songs, and removed duplicate songs. The output genres have been one-hot-encoded. Due to class imbalance, we experimented with oversampling and SMOTE techniques. Our GloVe embeddings were created using the Stanford NLP GitHub repository code, utilizing the corpus generated from our datasets.
    """)
    st.markdown("<h2 style='font-size: 30px;'>Data</h2>", unsafe_allow_html=True)
    st.write("""
    We worked with two primary datasets:
    - [Dataset 1: Music Dataset: 1950 to 2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019)
    - [Dataset 2: Song lyrics from 79 musical genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)

    In our preprocessing steps, we merged the data, dropped all non-English songs, and removed duplicate songs. The output genres were one-hot-encoded. Due to class imbalance, we experimented with oversampling and SMOTE techniques. Our GloVe embeddings were created using the Stanford NLP GitHub repository code, utilizing the corpus generated from our datasets.
    """) 

    st.markdown("<h2 style='font-size: 30px;'>Approach</h2>", unsafe_allow_html=True)

    # Approach details
    st.write("""
    The approach is to create GloVe embeddings from our datasets and use these embeddings in a logistic regression model, which serves as our baseline. We will then experiment using various classification algorithms to achieve an accurate model.
    """)

    st.markdown("<h2 style='font-size: 30px;'>Preprocessing</h2>", unsafe_allow_html=True)

   

    st.markdown("<h2 style='font-size: 30px;'>References</h2>", unsafe_allow_html=True)

    # References details with accessible links
    st.write("""
    - [Music Genre Classification using Song Lyrics](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report003.pdf)
    - [Train GloVe Embeddings using Stanford NLP code](https://stackoverflow.com/questions/48962171/how-to-train-glove-algorithm-on-my-own-corpus)
    - [Over 60,000 tracks are now uploaded to Spotify every day. that’s nearly one per second](https://www.musicbusinessworldwide.com/over-60000-tracks-are-now-uploaded-to-spotify-daily-thats-nearly-one-per-second/)
    - [Stanford NLP GitHub repository](https://github.com/stanfordnlp/GloVe/tree/master/eval)
    """)


def test_model_page():
    st.title("Testing the Model Page")
    
    # Text input box
    input_text = st.text_input("Enter text:", "")

    negation_words = {'against','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',
    "doesn't",'don', "don't",'hadn', "hadn't",'hasn', "hasn't",'haven', "haven't",'isn',
    "isn't",'mightn', "mightn't",'mustn', "mustn't",'needn', "needn't",'no','nor','not','shan', "shan't",'shouldn',
    "shouldn't",'wasn',"wasn't",'weren', "weren't","won't", 'wouldn', "wouldn't"}
    
    if input_text:
        # Clean and tokenize the input text
        cleaned_text = clean_text(input_text)
        cleaned_text_sentiment = clean_text_sentiment(input_text,negation_words)

        
        # Load GloVe word embeddings
        embedding_path = rootPath+"/vectors.txt"  # Change to the path of your GloVe embeddings file
        word_embeddings = load_glove_embeddings(embedding_path)
        
        # Get embedding for the input text
        input_embedding = get_lyric_embedding(cleaned_text, word_embeddings)
       # st.write("Predicted Genre Number:", input_embedding)
        
        # Load your trained model
        unzipFile(rootPath+'/best_model.pkl.zip')
        st.write(rootPath+'/best_model.pkl')
        gen_model = joblib.load(rootPath+'/best_model.pkl')
        # Predict genre number using the model
        genre_number = gen_model.predict(input_embedding.reshape(1, -1))[0]
        
        genre_names = {0: 'Hip Hop', 1: 'Pop', 2: 'Rock'}
        predicted_genre = genre_names.get(genre_number, 'Unknown Genre')
        st.write("Predicted Genre:", predicted_genre)
        # Perform sentiment analysis
        sentiments = getTopSentiments(cleaned_text_sentiment)
        
        # Display the sentiments
        st.write("Top Sentiments:", sentiments)

        if st.button("Add to Data"):
            # Create a DataFrame with the input text and predicted genre
            data = {
                "Artist": "xx",
                "target": "xx",
                "Lyric": input_text,
                "SName": "xx",
                "language": "xx",
                "cleaned_lyrics": cleaned_text,
                "Hip Hop": 1 if predicted_genre == "Hip Hop" else 0,
                "Pop": 1 if predicted_genre == "Pop" else 0,
                "Rock": 1 if predicted_genre == "Rock" else 0,
                "Genre": predicted_genre,
                "top_sentiments": sentiments
            }
            df = pd.DataFrame([data])
            
            # Append the DataFrame to the CSV file
            with open("/Users/roshanrayudu/Desktop/NLP_sem4/MusicClassficationNLP/data_with_sentiments.csv", "a") as file:
                df.to_csv(file, index=False, header=False)
    
   

def search_page():
    st.title("Search Page")
    
    # Load your CSV file
    @st.cache_resource
    def load_data():
        # Replace 'your_csv_file.csv' with the path to your CSV file
        df = pd.read_csv(rootPath+'/data_with_sentiments1.csv')
        return df
    df = load_data()
    
    # Genre filter
    selected_genre = st.multiselect("Select Genre", ["Hip Hop", "Pop", "Rock"])
    if selected_genre:
        df = df[df["Genre"].isin(selected_genre)]
    
    # Sentiment filter
    selected_sentiments = st.multiselect(
        "Select Sentiments", 
        ["anger 🤬", "disgust 🤢", "fear 😨", "joy 😀", "neutral 😐", "sadness 😭", "surprise 😲"]
    )
    if selected_sentiments:
        for sentiment in selected_sentiments:
            df = df[df["top_sentiments"].str.contains(sentiment.split()[0])]
    
    # Display filtered data
    selected_columns=['Artist','SName','Lyric','top_sentiments','Genre']
    if not df.empty:
        st.write(df[selected_columns].reset_index(drop=True))
    else:
        st.write("No data found matching the selected filters.")


def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Report", "Testing the Model", "Search"])
    
    if selected_page == "Report":
        report_page()
    elif selected_page == "Testing the Model":
        test_model_page()
    elif selected_page == "Search":
        search_page()

if __name__ == "__main__":
    main()


