import streamlit as st
import pickle

def test_model_page():
    st.title("Testing the Model Page")
    # Load your trained model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    # Add your code to test the model and display output
