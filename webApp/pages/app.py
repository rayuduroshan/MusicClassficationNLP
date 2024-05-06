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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

rootPath = str(Path(__file__).resolve().parent.parent)

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
    st.markdown("<h2 style='font-size: 35px;'>1. Introduction</h2>", unsafe_allow_html=True)

    # Main page content
    st.write("Our project's goal is to use song lyrics to categorize music into genres. Humans find it difficult to accomplish this task, and since borders are not always obvious, there is frequent discussion regarding where song fits in. Music genres show similarities between tracks, which helps to group music into collections. Songs often fit into more than one genre, indicating that genre isn't always clearly defined. Automating this classification process is highly motivated by technologies such as Spotify, which adds an estimated 60,000 songs to its database every day.")

   
    st.write("""
    We have merged the data, dropped all non-English songs, and removed duplicate songs. The output genres have been one-hot-encoded. Due to class imbalance, we experimented with oversampling and SMOTE techniques. Our GloVe embeddings were created using the Stanford NLP GitHub repository code, utilizing the corpus generated from our datasets.
    """)
    st.markdown("<h2 style='font-size: 35px;'>2. Data</h2>", unsafe_allow_html=True)
    st.write("""
    We worked with two primary datasets:
    - [Dataset 1: Music Dataset: 1950 to 2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019)
    - [Dataset 2: Song lyrics from 79 musical genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)
    
    """) 
    #images
     
     #Figire 1
     
    image_url = rootPath+"/pages/barplotOfGenreCounts.png"  # Replace this with your image URL
    st.image(image_url, caption='Figure 1.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">Bar graph representing the number of songs for each genre.</p>
</div>
""", unsafe_allow_html=True)
    
    #Figure 2
    image_url = rootPath+"/pages/wordcloud_['Hip Hop'].png"  # Replace this with your image URL
    st.image(image_url, caption='Figure 2.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">Wordcloud (Most frequently used words for Hip Hop) for Hip Hop.</p>
</div>
""", unsafe_allow_html=True)
    

    #Figure 3
    image_url = rootPath+"/pages/wordcloud_['Pop'].png"  # Replace this with your image URL
    st.image(image_url, caption='Figure 3.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">Wordcloud (Most frequently used words for Hip Hop) for Pop.</p>
</div>
""", unsafe_allow_html=True)
    
    #Figure 4
    image_url = rootPath+"/pages/wordcloud_['Rock'].png"  # Replace this with your image URL
    st.image(image_url, caption='Figure 4.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">Wordcloud(Most frequently used words for Hip Hop) for rock.</p>
</div>
""", unsafe_allow_html=True)
    


    st.markdown("<h2 style='font-size: 35px;'>3. Approach</h2>", unsafe_allow_html=True)

    # Approach details
    st.write("""
    The approach is to create GloVe embeddings from our datasets and use these embeddings in a logistic regression model, which serves as our baseline. We will then experiment using various classification algorithms to achieve an accurate model.
    """)

    st.markdown("<h3 style='font-size: 24px;'>3.1. Input Preprocessing</h3>", unsafe_allow_html=True)
    st.write("""
    1. Language Filtering: We identified and filtered out non-English songs from the dataset using the langdetect library, ensuring that our analysis focuses exclusively on English-language lyrics.
    2. Text Cleaning: The clean_text function was applied to the lyrics, which involved:
        - Converting text to lowercase for uniformity.
        - Removing punctuation marks to reduce noise.
        - Tokenizing the lyrics into individual words.
        - Removing stopwords (common words without significant meaning).
        - Lemmatizing tokens to their base forms for standardization.
    3. DataFrame Preparation: We created a refined DataFrame with essential columns such as artist information, genre labels, cleaned lyrics, and language identifiers, eliminating duplicates and missing values.
    4. GloVe Embeddings Loading:  
        - The get_word_embedding function retrieves the embedding vector for a given word from the loaded GloVe embeddings. If the word is not found, it returns a zero vector.
        - The get_lyric_embedding function tokenizes lyrics into words and calculates the average embedding vector for the entire lyric by averaging the embeddings of individual words.
        - The code applies the get_lyric_embedding function to each row in the DataFrame df['cleaned_lyrics'], creating a new column lyric_embedding containing the computed lyric embeddings.       
    5. Outcome: The preprocessing yielded a clean and structured dataset suitable for subsequent analyses
    """) 
 
    st.markdown("<h3 style='font-size: 24px;'>3.2. Outputs</h3>", unsafe_allow_html=True)
    st.write(""" 
        1. MultiLabelBinarizer Encoding for Genre Labels
             - Encoding Genre Labels: MultiLabelBinarizer is instantiated to encode the 'target' column (genre labels) into binary form, creating separate columns for each genre label.
             - Creating Encoded DataFrame: The encoded targets are transformed into a DataFrame with binary columns representing each unique genre label.
             - The resulting DataFrame contains binary-encoded genre labels, enabling genre-specific analyses and machine learning tasks. The number of unique genre classes and the total count of English songs in the dataset are printed for reference.
            
             """)
    
    st.markdown("<h3 style='font-size: 24px;'>3.3. Classification</h3>", unsafe_allow_html=True)
    st.write("""
             1. Baseline Models
                    - The baseline models were initially trained and evaluated on the imbalanced dataset to assess their performance in the natural data distribution.
                    - At the outset of the analysis, Logistic Regression, Random Forest, and XGBoost were selected as baseline classification models for music genre prediction based on lyrics data. The dataset exhibited a notable class imbalance, with 'Pop' being heavily represented, followed by 'Rock' and 'Hip Hop' with fewer instances
             """)
    image_url = rootPath+"/pages/cm_without_class_balance.png" 
    st.image(image_url, caption='Figure 5.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">cm_without_class_balance.</p>
</div>
""", unsafe_allow_html=True)
    st.write("""        
              2. Handling Class Imbalance
                    - Through the implementation of SMOTE oversampling, the project successfully managed class imbalance and improved the robustness of machine learning models for music genre classification based on lyrics data.
            """)
    
    image_url = rootPath+"/pages/logistic_after_balance.png" 
    st.image(image_url, caption='Figure 6.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">logistic_after_balance</p>
</div>
""", unsafe_allow_html=True)
    st.write("""         3. Best Model
                    - Random Forest's combination of ensemble learning principles, robustness, scalability, interpretability, and feature importance analysis makes it the best model for music genre classification tasks based on lyrics data. Its ability to handle complex relationships, generalize to new data, and provide valuable insights into genre characteristics positions Random Forest as a top-performing and reliable choice for this domain.

""")
    image_url = rootPath+"/pages/various_f1scores.png" 
    st.image(image_url, caption='Figure 7.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">F1 scores</p>
</div>
""", unsafe_allow_html=True)
    


    image_url = rootPath+"/pages/best_model.png" 
    st.image(image_url, caption='Figure 8.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">Best model.</p>
</div>
""", unsafe_allow_html=True)
    

    st.markdown("<h3 style='font-size: 24px;'>3.4. Sentiment Analysis</h3>", unsafe_allow_html=True)
    st.write("""
    We have performed Sentiment Analysis on our lyrics dataset using a Bert based model (j-hartmann/emotion-english-distilroberta-base). We chose this model because bert models use attention mechanisms to better capture the context and dependencies between words and are better at handling long sequences. This model has been trained on 6 different datasets and has been finetuned to predict scores for 6 different emotions and a neutral class.
We use this model to get scores for 7 classes which are anger, disgust, fear, joy, neutral, sadness and surprise. Once we get emotion scores for song, we assign top three emotions to that song using 0.2 as a threshold (we assign atmost three different emotions to a song, only assign an emotion to a song from the top 3 emotion scores if score is greater than 0.2).
""")

    st.markdown("<h3 style='font-size: 24px;'>3.5. Results</h3>", unsafe_allow_html=True)

    image_url = rootPath+"/pages/hiphop.png" 
    st.image(image_url, caption='Figure 9.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">HipHop</p>
</div>
""", unsafe_allow_html=True)
    

    image_url = rootPath+"/pages/pop.png" 
    st.image(image_url, caption='Figure 10.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">Pop.</p>
</div>
""", unsafe_allow_html=True)
    

    image_url = rootPath+"/pages/rock.png" 
    st.image(image_url, caption='Figure 11.', use_column_width=True)
    st.markdown("""
<div style="text-align: center;">
    <p style="font-style: italic;">Rock.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<h2 style='font-size: 35px;'>References</h2>", unsafe_allow_html=True)

    # References details with accessible links
    st.write("""
    - [Music Genre Classification using Song Lyrics](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report003.pdf)
    - [Train GloVe Embeddings using Stanford NLP code](https://stackoverflow.com/questions/48962171/how-to-train-glove-algorithm-on-my-own-corpus)
    - [Over 60,000 tracks are now uploaded to Spotify every day. that’s nearly one per second](https://www.musicbusinessworldwide.com/over-60000-tracks-are-now-uploaded-to-spotify-daily-thats-nearly-one-per-second/)
    - [Stanford NLP GitHub repository](https://github.com/stanfordnlp/GloVe/tree/master/eval)
    - [Hugging face.co](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
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
        song_name = st.text_input("Enter Song Name:", "")
        artist_name = st.text_input("Enter Artist Name:", "")
        if st.button("Add Song to Database"):
            # Create a DataFrame with the input text and predicted genre
            data = {
                "Artist": artist_name,
                "target": "xx",
                "Lyric": input_text,
                "SName": song_name,
                "language": "xx",
                "cleaned_lyrics": cleaned_text,
                "Hip Hop": 1 if predicted_genre == "Hip Hop" else 0,
                "Pop": 1 if predicted_genre == "Pop" else 0,
                "Rock": 1 if predicted_genre == "Rock" else 0,
                "top_sentiments": sentiments,
                "Genre": predicted_genre
            }
            df = pd.DataFrame([data])
            # Append the DataFrame to the CSV file
            with open(rootPath+"/data_with_sentiments1.csv", "a") as file:
                df.to_csv(file, index=False, header=False)
            st.cache_resource.clear()
    
   

def search_page():
    st.title("Search Page")
    
    # Load your CSV file
    @st.cache_resource
    def load_data():
        # Replace 'your_csv_file.csv' with the path to your CSV file
        df = pd.read_csv(rootPath+'/data_with_sentiments1.csv')
        return df
    df = load_data()
    song_name_text = st.text_input("Enter Song Name:", "")
    if song_name_text:
        df = df[df["SName"].str.lower().str.contains(song_name_text.lower(), na=False)]
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


