# Automated Tagging of music data based on lyrics using Natural language processing.

## Introduction

Our project's goal is to use song lyrics to categorize music into genres. Humans find it difficult to accomplish this task, and since borders are not always obvious, there is frequent discussion regarding where song fits in. Music genres show similarities between tracks, which helps to group music into collections. Songs often fit into more than one genre, indicating that genre isn't always clearly defined. Automating this classification process is highly motivated by technologies such as Spotify, which adds an estimated 60,000 songs to its database every day.

## Approach

The approach is to create GloVe embedding from our datasets and use these embeddings in a logistic regression model which serves as our baseline and then experiment using classification algorithms to achieve an accurate model.

### Preprocessing
 We have merged the data, dropped all the non-English songs, and removed duplicate songs. We have one-hot-encoded the output genres. Since there is a class imbalance, we experimented with oversampling and SMOTE techniques to handle this imbalance. We've created our own GloVe embeddings using the code from the Stanford NLP GitHub repository, utilizing the corpus generated from our datasets.

## Data

1. Dataset 1: [Music Dataset: 1950 to 2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019?rvi=1)
2. Dataset 2: [Song lyrics from 79 musical genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres?rvi=1)

## Web-based UI:
We've created a user-friendly web interface using the Streamlit framework in Python. Our interface allows users to test a model by inputting data, searching and filtering songs based on genres and emotions, and even adding new data. It provides an intuitive platform for exploring music based on various criteria, enhancing the user experience with streamlined functionality.

[MusicClassificationUsingLyrics](https://music-classfication-lyrics.streamlit.app/)

## References

1. [Music Genre Classification using Song Lyrics](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report003.pdf)
2. [Train Glove Embeddings using Stanford NLP code](https://stackoverflow.com/questions/48962171/how-to-train-glove-algorithm-on-my-own-corpus)
3. [Over 60,000 tracks are now uploaded to Spotify every day. that’s nearly one per second](https://www.musicbusinessworldwide.com/over-60000-tracks-are-now-uploaded-to-spotify-daily-thats-nearly-one-per-second/)
4. [Stanford NLP GitHub repository](https://github.com/stanfordnlp/GloVe/tree/master/eval)
