import pandas as pd
import openai
import json
import logging
import tiktoken
import numpy as np

tokenizer = tiktoken.get_encoding("cl100k_base")

from embedding_utils import get_max_num_tokens, get_embeddings, get_doc_model

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, doc_embeddings: pd.DataFrame):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_model = get_doc_model()
    query_embedding = get_embeddings(query, model=query_model)
    doc_embeddings['similarity'] = doc_embeddings['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    doc_embeddings.sort_values(by='similarity', inplace=True, ascending=False)
    doc_embeddings.reset_index(drop=True, inplace=True)
    
    return doc_embeddings
def construct_prompt(query, doc_embeddings):
    MAX_SECTIONS = 10 
    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n---\n"
    separator_len = len(tokenizer.encode(SEPARATOR))

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_links = []

    for section_index in range(len(doc_embeddings)):
        document_section = doc_embeddings.loc[section_index]

        # Access metadata_filename using column name
        metadata_filename = document_section['metadata_filename']

        # Load metadata from JSON file with error handling
        try:
            with open(metadata_filename, 'r') as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading metadata for page '{document_section.title}': {e}")
            continue  # Skip this section if there's an error loading metadata

        body = metadata['body']

        # Reintroduce token limit handling and potentially improve truncation logic
        while len(tokenizer.encode(body)) > MAX_SECTION_LEN:
            # Truncate to the nearest sentence or paragraph boundary (you'll need to implement this)
            body = body[:-10]  

        # Include relevant metadata in the prompt (customize as needed)
        chosen_sections.append(f"### Title: {metadata['title']} | Created by: {metadata['creator']} | Last Updated: {metadata['last_updated']} | Body: {body}")
        chosen_sections_links.append(document_section.link)

        chosen_sections_len += len(tokenizer.encode(body)) + separator_len
        if chosen_sections_len > MAX_SECTION_LEN or len(chosen_sections) >= MAX_SECTIONS:
            break

    header = """Use the following Confluence excerpts to answer the question. If the answer cannot be found in the provided context, say "I don't know."\n\n"""

    prompt = header + SEPARATOR.join(chosen_sections) + f"\n\nQuestion: {query}\nAnswer:"

    return (prompt, chosen_sections_links)

def internal_doc_chatbot_answer(query, DOC_title_content_embeddings):
    
    # Order docs by similarity of the embeddings with the query
    DOC_title_content_embeddings = order_document_sections_by_query_similarity(query, DOC_title_content_embeddings)
    # Construct the prompt
    prompt, links = construct_prompt(query, DOC_title_content_embeddings)
    # Ask the question with the context to ChatGPT
    COMPLETIONS_MODEL = "gpt-3.5-turbo"  # Or another chat-optimized model

    response = openai.ChatCompletion.create(  # Use ChatCompletion instead of Completion
        model=COMPLETIONS_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}  # Pass the prompt as the user message
        ],
        temperature=0, 
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    output = response.choices[0].message['content'].strip(" \n")  # Access the content from the message

    return output, links