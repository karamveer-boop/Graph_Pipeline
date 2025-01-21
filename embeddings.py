import os
import pandas as pd
import openai
from .utils import truncate_text

# openai.api_key = "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"
# def generate_embedding(text):
#     """
#     Generate embeddings using OpenAI's API.
#     """
#     if not isinstance(text, str) or not text.strip():
#         return None  # Skip invalid text

#     # Truncate text to ensure it fits within token limits
#     text = truncate_text(text, max_chars=2000)

#     response = openai.Embedding.create(
#         model="text-embedding-ada-002",
#         input=text
#     )
#     return response['data'][0]['embedding']

import requests

VOYAGE_API_URL = "https://api.voyage-3.com/embed"
VOYAGE_API_KEY = "pa-9hpDOXO1xsJWxjJomc-GQhEwTBpmPsKqG4kd-SLuaro"  # Replace with your API key

def generate_embedding(text):
    """
    Generate embeddings using Voyage-3-large API.
    """
    if not isinstance(text, str) or not text.strip():
        return None  # Skip invalid text
    
    payload = {
        "model": "voyage-3-large",
        "input": text
    }
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(VOYAGE_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("embedding")
    else:
        print(f"Error with Voyage API: {response.status_code} - {response.text}")
        return None


# def process_chunk_embeddings(input_dir, output_dir):
#     """
#     Process chunks to generate embeddings and save them.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     for chunk_file in os.listdir(input_dir):
#         if chunk_file.endswith('.csv'):
#             try:
#                 # Explicitly specify encoding and handle decoding errors
#                 data = pd.read_csv(os.path.join(input_dir, chunk_file), encoding='utf-8', on_bad_lines='skip')
#             except UnicodeDecodeError:
#                 print(f"Failed to decode {chunk_file}. Skipping.")
#                 continue
            
#             # Combine relevant columns into content
#             # Modify the 'process_chunk_embeddings' function in embeddings.py
#             data['content'] = (
#                 data['Shloka Content'].astype(str) + " " +
#                 data['Hindi Translation Of Sri Shankaracharya Sanskrit Commentary By Sri Harikrishnadas Goenka'].astype(str) + " " +
#                 data['English Translation by Shri Purohit Swami'].astype(str)
#             )

#             # Add metadata columns directly for later use
#             data['Language'] = data['Language'].fillna('N/A')
#             data['Chapter'] = data['Chapter'].fillna('N/A')
#             data['Shloka'] = data['Shloka'].fillna('N/A')
#             data['Shloka Content'] = data['Shloka Content'].fillna('N/A')
#             data['Hindi Translation'] = data['Hindi Translation Of Sri Shankaracharya Sanskrit Commentary By Sri Harikrishnadas Goenka'].fillna('N/A')
#             data['English Translation'] = data['English Translation by Shri Purohit Swami'].fillna('N/A')

            
#             # Clean and truncate content
#             data['content'] = data['content'].apply(
#                 lambda x: truncate_text(
#                     ''.join(c for c in x if c.isprintable()).strip(),
#                     max_chars=2000
#                 )
#             )
            
#             # Filter out empty or invalid content
#             valid_data = data[data['content'].str.strip().astype(bool)]
            
#             # Generate embeddings and log errors
#             embeddings = []
#             for i, row in valid_data.iterrows():
#                 try:
#                     embedding = generate_embedding(row['content'])
#                     if embedding:
#                         embeddings.append(embedding)
#                     else:
#                         raise ValueError("Empty embedding")
#                 except Exception as e:
#                     print(f"Error generating embedding for row {i} in {chunk_file}: {e}")
#                     embeddings.append(None)
            
#             valid_data['embedding'] = embeddings
            
#             # Save embeddings with metadata
#             valid_data.to_json(
#                 os.path.join(output_dir, f"{chunk_file.split('.')[0]}_embeddings.json"),
#                 orient='records',
#                 force_ascii=False  # Preserve non-ASCII characters
#             )
#     print(f"Embeddings saved to {output_dir}.")
def process_chunk_embeddings(input_dir, output_dir):
    """
    Process chunks to generate embeddings for entire rows and save them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for chunk_file in os.listdir(input_dir):
        if chunk_file.endswith('.csv'):
            try:
                # Read CSV file with UTF-8 encoding
                data = pd.read_csv(os.path.join(input_dir, chunk_file), encoding='utf-8', on_bad_lines='skip')
            except Exception as e:
                print(f"Failed to load {chunk_file}: {e}")
                continue
            
            # Combine all columns into a single string for embedding
            data['content'] = data.apply(
                lambda row: ' '.join([f"{col}: {str(row[col])}" for col in row.index if pd.notnull(row[col])]),
                axis=1
            )
            
            # Generate embeddings for each row
            embeddings = []
            for i, row in data.iterrows():
                try:
                    embedding = generate_embedding(row['content'])
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        raise ValueError("Empty embedding")
                except Exception as e:
                    print(f"Error generating embedding for row {i} in {chunk_file}: {e}")
                    embeddings.append(None)
            
            data['embedding'] = embeddings
            
            # Save embeddings with metadata
            output_path = os.path.join(output_dir, f"{chunk_file.split('.')[0]}_embeddings.json")
            data.to_json(output_path, orient='records', force_ascii=False)
            print(f"Saved embeddings for {chunk_file} to {output_path}.")
