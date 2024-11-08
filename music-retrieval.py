import streamlit as st
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import time

from utils import bm25okapi_search
import preprocess

st.title("SONG'S INFORMATION RETRIEVAL FROM LYRICS")
uploaded_data_file = st.file_uploader("Choose a music data file")
if uploaded_data_file is not None:
    df = pd.read_csv(uploaded_data_file, encoding='utf-8')

# Preprocessing pipeline setup
pipeline = ["PorterStemmer", "remove_stopword", "nltk_word_tokenizer"]

# Sidebar button to start preprocessing
if 'tokenize_lyric' not in st.session_state:
    st.session_state.tokenize_lyric = None
if 'preprocesser' not in st.session_state:
    st.session_state.preprocesser = None

if st.sidebar.button('Start Preprocessing', disabled=(uploaded_data_file is None)):
    lyrics = np.array([])
    preprocesser = preprocess.Preprocessing(Pipeline=pipeline)
    process_bar = st.sidebar.progress(0)
    with st.spinner(text="Please wait..."):
        start = time.time()
        for i, lyric in enumerate(df.lyrics):
            lyrics = np.append(lyrics, preprocesser.Preprocess(lyric))
            process_bar.progress((i + 1) / len(df.lyrics))
        end = time.time()
        tokenize_lyric = [lyric.split() + [f"-->{df.id[i]}"] for i, lyric in enumerate(lyrics)]

    st.session_state.tokenize_lyric = tokenize_lyric
    st.session_state.preprocesser = preprocesser
    st.session_state.time = end - start

if st.session_state.tokenize_lyric is not None:
    st.sidebar.success('Preprocessing Done!', icon="✅")
    st.sidebar.write(f'Execution time: {st.session_state.time:.4f}s')

# Search functionality
st.header("Search Engine")
query = st.text_input('Input query:')
top_relevant = st.number_input('Number of most relevant results', min_value=1, max_value=100)

if st.button("Search"):
    if st.session_state.tokenize_lyric is None:
        st.error("You haven't preprocessed the data yet! Please run it first!", icon="🚨")
    else:
        preprocesser = st.session_state.preprocesser
        token = preprocesser.Preprocess(query.replace('...', ' ')).split(" ")
        with st.spinner(text="Searching..."):
            tokenize_lyric = st.session_state.tokenize_lyric
            bm25 = BM25Okapi(tokenize_lyric)
            rank = bm25okapi_search(token, bm25, tokenize_lyric, n_results=top_relevant)
        st.write(df[df.id.isin(rank)].reset_index(drop=True))
