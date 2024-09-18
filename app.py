import streamlit as st
import logging
import sys
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("openai-api-key")

from confluence_utils import connect_to_Confluence, get_spaces
from embedding_utils import return_Confluence_embeddings, get_token_count
from prompt_utils import internal_doc_chatbot_answer

# Streamlit App Title
st.title("Confluence Chatbot with GPT-3.5-turbo Tokenizer")

# Get Confluence spaces
confluence = connect_to_Confluence()
if confluence is None:
    st.error("Failed to connect to Confluence. Please check your credentials.")
    sys.exit(1)

spaces = get_spaces(confluence)

# Space selection
selected_space_key = st.selectbox("Select Confluence Space", [space['key'] for space in spaces])

# Chat history management
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        st.write(f"**User:** {message['user']}")
        st.write(f"**Bot:** {message['bot']}\n")

# User input
query = st.text_area("Enter your query:", "")
st.write(f"Query token count: {get_token_count(query)}")

if st.button("Get Chatbot Response"):
    logging.info("User clicked 'Get Chatbot Response' button")

    # Fetch embeddings 
    if 'DOC_title_content_embeddings' not in st.session_state: 
        with st.spinner("Fetching Confluence embeddings..."):
            try:
                st.session_state.DOC_title_content_embeddings = return_Confluence_embeddings(selected_space_key)
            except Exception as e:
                logging.error(f"Error fetching Confluence embeddings: {e}", exc_info=True) 
                st.error(f"Error fetching Confluence embeddings: {e}")

    DOC_title_content_embeddings = st.session_state.DOC_title_content_embeddings

    output, links = internal_doc_chatbot_answer(query, DOC_title_content_embeddings)

    st.session_state.chat_history.append({
        'user': query,
        'bot': output
    })

    # Redisplay chat history
    with chat_container:
        for message in st.session_state.chat_history:
            st.write(f"**User:** {message['user']}")
            st.write(f"**Bot:** {message['bot']}\n")
            
    if links:
        st.write("Relevant Links:")
        for link in links:
            st.markdown(f"- [{link}]({link})")