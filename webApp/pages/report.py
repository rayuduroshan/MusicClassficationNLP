import streamlit as st


    # Main page title
st.title("Report Page")

    # Side heading using Markdown syntax
    # st.markdown("## Introduction")
st.markdown("<h2 style='font-size: 30px;'>Introduction</h2>", unsafe_allow_html=True)

    # Main page content
st.write("Our project's goal is to use song lyrics to categorize music into genres. Humans find it difficult to accomplish this task, and since borders are not always obvious, there is frequent discussion regarding where song fits in. Music genres show similarities between tracks, which helps to group music into collections. Songs often fit into more than one genre, indicating that genre isn't always clearly defined. Automating this classification process is highly motivated by technologies such as Spotify, which adds an estimated 60,000 songs to its database every day.")

st.markdown("<h2 style='font-size: 30px;'>Approach</h2>", unsafe_allow_html=True)

    # Approach details
st.write("""
    The approach is to create GloVe embeddings from our datasets and use these embeddings in a logistic regression model, which serves as our baseline. We will then experiment using various classification algorithms to achieve an accurate model.
    """)

st.markdown("<h2 style='font-size: 30px;'>Preprocessing</h2>", unsafe_allow_html=True)

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

st.markdown("<h2 style='font-size: 30px;'>References</h2>", unsafe_allow_html=True)

    # References details with accessible links
st.write("""
    - [Music Genre Classification using Song Lyrics](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report003.pdf)
    - [Train GloVe Embeddings using Stanford NLP code](https://stackoverflow.com/questions/48962171/how-to-train-glove-algorithm-on-my-own-corpus)
    - [Over 60,000 tracks are now uploaded to Spotify every day. that’s nearly one per second](https://www.musicbusinessworldwide.com/over-60000-tracks-are-now-uploaded-to-spotify-daily-thats-nearly-one-per-second/)
    - [Stanford NLP GitHub repository](https://github.com/stanfordnlp/GloVe/tree/master/eval)
    """)

