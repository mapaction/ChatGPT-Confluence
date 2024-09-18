import tiktoken
import pandas as pd
import numpy as np
import datetime
import os
import json
import logging
import nltk
import sys
import atlassian
import traceback
from bs4 import BeautifulSoup
import openai

from confluence_utils import get_all_pages, get_url, connect_to_Confluence
from openai_utils import get_doc_model, get_embeddings

# Initialize tokenizer (consider moving this to a separate utils file if needed elsewhere)
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_token_count(text):
  """Returns the number of tokens in a text string."""
  return len(tokenizer.encode(text))

def get_max_num_tokens():
    return 2046

def collect_title_body_embeddings(pages, space, save_csv=True):
    """
    From a list of page objects, get the title and the body, calculate
    the number of tokens as well as the embeddings of the body. Also creates
    metadata JSON files for each page.

    Parameters
    ----------
    pages: List of page objects, i.e. output of get_all_pages()
    save_csv: Boolean. If True, the dataframe is saved locally into a CSV file.

    Return
    ------
    A dataframe of the title, body, metadata filename, and other details of all pages.
    """

    collect = []
    for page in pages:
        title = page['title']

        # Error handling for get_url
        base_url = get_url()
        if base_url is None:
            logging.error(f"Error: Unable to construct link for page '{title}' due to missing base URL.")
            link = "URL not available" 
        else:
            link = base_url + '/spaces/' + space + '/pages/' + page['id']

        htmlbody = page['body']['storage']['value']
        htmlParse = BeautifulSoup(htmlbody, 'html.parser')
        body = []

        # Extract additional information with error handling
        try:
            creator = page['history']['createdBy']['displayName']
            created_date = page['history']['createdDate']
            last_updated = page['history']['lastUpdated']['when']
        except KeyError:
            print(f"Warning: 'history' attribute not found for page '{title}'. Using default values.")
            creator = "Unknown"
            created_date = "Unknown"
            last_updated = "Unknown"

        views = page['views']

        # Filter sentences based on POS tags 
        for para in htmlParse.find_all("p"):
            sentence = para.get_text()
            tokens = nltk.tokenize.word_tokenize(sentence)
            token_tags = nltk.pos_tag(tokens)
            tags = [x[1] for x in token_tags]
            if any([x[:2] == 'VB' for x in tags]) and any([x[:2] == 'NN' for x in tags]):  # Check for at least one verb and one noun
                body.append(sentence)
        body = '. '.join(body)

        # Truncate the body if it exceeds the token limit
        while get_token_count(body) > get_max_num_tokens():
            body = body[:-10]  # Remove the last 500 characters and retry
            # You might want to refine this truncation to preserve sentence boundaries

        # Calculate number of tokens after truncation
        num_tokens = get_token_count(body)

        # Create metadata dictionary
        metadata = {
            'title': title,
            'id': page['id'],
            'space_key': space,
            'url': link,
            'creator': creator,
            'created_date': created_date,
            'last_updated': last_updated,
            'views': views,
            'body': body,
            'num_tokens': num_tokens
        }

        # Save metadata to JSON file
        metadata_filename = f"{page['id']}_metadata.json"

        print(f"Processing page: {title}")
        print(f"Metadata filename: {metadata_filename}")
        print(f"Metadata: {metadata}")

        logging.info(f"Metadata saved for page '{title}' to {metadata_filename}")
        
        collect += [(title, link, body, num_tokens, creator, created_date, last_updated, views, metadata_filename)]

        try:
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=4)
            logging.info(f"Metadata saved for page '{title}' to {metadata_filename}")
        except (IOError, OSError) as e:
            logging.error(f"Error saving metadata for page '{title}': {e}")
        except (TypeError, ValueError) as e:
            logging.error(f"Error serializing metadata for page '{title}': {e}")



    # Create DataFrame 
    DOC_title_content_embeddings = pd.DataFrame(collect, columns=[
        'title', 'link', 'body', 'num_tokens', 'creator', 'created_date', 'last_updated', 'views', 'metadata_filename'
    ])

    # Print DataFrame columns for debugging
    print(DOC_title_content_embeddings.columns)
    # Calculate embeddings (only for rows within token limit)
    DOC_title_content_embeddings = DOC_title_content_embeddings[DOC_title_content_embeddings.num_tokens <= get_max_num_tokens()]
    doc_model = get_doc_model()

    # Error handling for OpenAI API call
    try:
        DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings.body.apply(lambda x: get_embeddings(x, doc_model))
    except openai.error.OpenAIError as e:
        logging.error(f"Error calculating embeddings: {e}")

    if save_csv:
        try:
            DOC_title_content_embeddings.to_csv('./DOC_title_content_embeddings.csv', index=False)
            print("CSV file 'DOC_title_content_embeddings.csv' saved successfully.")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

    return DOC_title_content_embeddings

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def update_internal_doc_embeddings(space_key):
    confluence = connect_to_Confluence()

    # get space from front end
    if confluence is None:  # Check if the connection was successful
        print("Failed to connect to Confluence. Exiting...")
        sys.exit(1)

    # Get page contents
    pages = get_all_pages(confluence, space_key)

    # Debugging: Print the number of pages retrieved
    print(f"Retrieved {len(pages)} pages from Confluence.")

    # Debugging: Print the titles of the retrieved pages
    for page in pages:
        print(f"- {page['title']}")

    # Extract title, body and number of tokens
    DOC_title_content_embeddings = collect_title_body_embeddings(pages, space_key, save_csv=True)
    return DOC_title_content_embeddings

def parse_numbers(s):
  return [float(x) for x in s.strip('[]').split(',')]

def return_Confluence_embeddings(space_key):
    """Retrieves Confluence embeddings from a CSV file or generates new ones if needed."""

    today = datetime.datetime.today()  # Get today's date
    confluence_embeddings_file = 'DOC_title_content_embeddings.csv'  # File to store embeddings

    try:
        # Check if file exists and is recent enough
        if os.path.exists(confluence_embeddings_file):
            confluence_embeddings_file_date = datetime.datetime.fromtimestamp(
                os.path.getmtime(confluence_embeddings_file)
            )
            delta = today - confluence_embeddings_file_date
            if delta.days <= 7:  # If file is less than a week old, read from it
                doc_title_content_embeddings = pd.read_csv(confluence_embeddings_file, dtype={'embeddings': object})
                doc_title_content_embeddings['embeddings'] = doc_title_content_embeddings['embeddings'].apply(parse_numbers)
                return doc_title_content_embeddings

        # If file doesn't exist or is too old, generate new embeddings
        print("Generating new Confluence embeddings...")
        doc_title_content_embeddings = update_internal_doc_embeddings(space_key)  # Pass space_key
        return doc_title_content_embeddings

    except atlassian.errors.ApiPermissionError:
        print("Error: The Confluence user does not have permission to view the content.")
        # You might also want to return an empty DataFrame or handle the error in a more appropriate way
        sys.exit(1)

    except FileNotFoundError:
        print("Error: 'DOC_title_content_embeddings.csv' not found. Creating new embeddings...")
        doc_title_content_embeddings = update_internal_doc_embeddings()
        return doc_title_content_embeddings

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        # Handle other potential exceptions or re-raise them if necessary
        sys.exit(1)