# Automated Tagging of music data based on lyrics using Natural language processing.

## Introduction

Our project's goal is to use song lyrics to categorize music into genres. Humans find it difficult to accomplish this task, and since borders are not always obvious, there is frequent discussion regarding where song fits in. Music genres show similarities between tracks, which helps to group music into collections. Songs often fit into more than one genre, indicating that genre isn't always clearly defined. Automating this classification process is highly motivated by technologies such as Spotify, which adds an estimated 60,000 songs to its database every day.

## Web-based UI:
We've created a user-friendly web interface using the Streamlit framework in Python. Our interface allows users to test a model by inputting data, searching and filtering songs based on genres and emotions, and even adding new data. It provides an intuitive platform for exploring music based on various criteria, enhancing the user experience with streamlined functionality.

Click here for website : [MusicClassificationUsingLyrics](https://music-classfication-lyrics.streamlit.app/)

## Data

1. Dataset 1: [Music Dataset: 1950 to 2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019?rvi=1)
2. Dataset 2: [Song lyrics from 79 musical genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres?rvi=1)

| ![Bar graph representing the number of songs for each genre](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/barplotOfGenreCounts.png) | 
|:--:| 
| *Bar graph representing the number of songs for each genre* |

  | ![Wordcloud (Most frequently used words for Hip Hop) for Hip Hop](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/wordcloud_%5B'Hip%20Hop'%5D.png) | 
|:--:| 
| *Wordcloud (Most frequently used words for Hip Hop) for Hip Hop* |

|  ![Wordcloud (Most frequently used words for Pop) for Pop](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/wordcloud_%5B'Pop'%5D.png) | 
|:--:| 
| *Wordcloud (Most frequently used words for Pop) for Pop* |

| ![Bar graph representing the number of songs for each genre](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/wordcloud_%5B'Rock'%5D.png) | 
|:--:| 
| *Wordcloud (Most frequently used words for Rock) for Rock* |
  
 
  
   

## Approach

The approach is to create GloVe embeddings from our datasets and use these embeddings in a logistic regression model, which serves as our baseline. We will then experiment using various classification algorithms to achieve an accurate model.

### Input Preprocessing

1. **Language Filtering**:
   - We identified and filtered out non-English songs from the dataset using the langdetect library, ensuring that our analysis focuses exclusively on English-language lyrics.  
2. **Text Cleaning**:
   - The `clean_text` function was applied to the lyrics, which involved:
     - Converting text to lowercase for uniformity.
     - Removing punctuation marks to reduce noise.
     - Tokenizing the lyrics into individual words.
     - Removing stopwords (common words without significant meaning).
     - Lemmatizing tokens to their base forms for standardization.  
3. **DataFrame Preparation**:
   - We created a refined DataFrame with essential columns such as artist information, genre labels, cleaned lyrics, and language identifiers, eliminating duplicates and missing values.   
4. **GloVe Embeddings Loading**:
   - The `get_word_embedding` function retrieves the embedding vector for a given word from the loaded GloVe embeddings. If the word is not found, it returns a zero vector.
   - The `get_lyric_embedding` function tokenizes lyrics into words and calculates the average embedding vector for the entire lyric by averaging the embeddings of individual words.
   - The code applies the `get_lyric_embedding` function to each row in the DataFrame `df['cleaned_lyrics']`, creating a new column `lyric_embedding` containing the computed lyric embeddings.
 5. Outcome: The preprocessing yielded a clean and structured dataset suitable for subsequent analyses

## Output Preprocessing

### MultiLabelBinarizer Encoding for Genre Labels
1. **Encoding Genre Labels**:
   - MultiLabelBinarizer is instantiated to encode the 'target' column (genre labels) into binary form, creating separate columns for each genre label.
2. **Creating Encoded DataFrame**:
   - The encoded targets are transformed into a DataFrame with binary columns representing each unique genre label.   
3. **Resulting DataFrame**:
   - The resulting DataFrame contains binary-encoded genre labels, enabling genre-specific analyses and machine learning tasks.    

## Classification

1. **Baseline Models Selection**:
   - The baseline models were initially trained and evaluated on the imbalanced dataset to assess their performance in the natural data distribution.
   - Logistic Regression is the selected baseline classification models for music genre prediction based on lyrics data. The dataset exhibited a notable class imbalance, with 'Pop' being heavily represented, followed by 'Rock' and 'Hip Hop' with fewer instances
  
| ![cm_without_class_balance](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/cm_without_class_balance.png) | 
|:--:| 
|cm_without_class_balance|

2. **SMOTE Oversampling**:
   - Through the implementation of SMOTE oversampling, the project successfully managed class imbalance and improved the robustness of machine learning models for music genre 
     classification based on lyrics data.

| ![cm_without_class_balance](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/logistic_after_balance.png) | 
|:--:|
|after class balance|

3. **Random Forest**:
   - Random Forest's combination of ensemble learning principles, robustness, scalability, interpretability, and feature importance analysis makes it the best model for music genre 
     classification tasks based on lyrics data. Its ability to handle complex relationships, generalize to new data, and provide valuable insights into genre characteristics positions 
     Random Forest is a top-performing and reliable choice for this domain.
     
| ![cm_without_class_balance](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/various_f1scores.png) | 
|:--:|
|various F1 scores|

| ![cm_without_class_balance](https://github.com/rayuduroshan/MusicClassficationNLP/blob/main/webApp/pages/best_model.png) | 
|:--:|
|best model : Random Forest|

## Sentiment Analysis

### Using Bert-based Model
- Sentiment Analysis has been performed on our lyrics dataset using a Bert-based model ([j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)).
- We chose this model because Bert models use attention mechanisms to better capture the context and dependencies between words and are better at handling long sequences.
- The model has been trained on 6 different datasets and has been fine-tuned to predict scores for 6 different emotions and a neutral class.
- We use this model to get scores for 7 classes which are anger, disgust, fear, joy, neutral, sadness, and surprise.
- Once we get emotion scores for a song, we assign the top three emotions to that song using 0.2 as a threshold. We assign at most three different emotions to a song, only assigning an emotion to a song from the top 3 emotion scores if the score is greater than 0.2.


## References

1. [Music Genre Classification using Song Lyrics](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report003.pdf)
2. [Train Glove Embeddings using Stanford NLP code](https://stackoverflow.com/questions/48962171/how-to-train-glove-algorithm-on-my-own-corpus)
3. [Over 60,000 tracks are now uploaded to Spotify every day. that’s nearly one per second](https://www.musicbusinessworldwide.com/over-60000-tracks-are-now-uploaded-to-spotify-daily-thats-nearly-one-per-second/)
4. [Stanford NLP GitHub repository](https://github.com/stanfordnlp/GloVe/tree/master/eval)
