import os
import json
import pickle
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# def build_graph(embedding_dir, output_file, similarity_threshold=0.8):
#     """
#     Build a graph from embeddings and save it.
#     """
#     graph = nx.Graph()

#     # Load embeddings
#     for embedding_file in os.listdir(embedding_dir):
#         if embedding_file.endswith('.json'):
#             with open(os.path.join(embedding_dir, embedding_file), 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             # Filter out rows with invalid embeddings
#             valid_data = [row for row in data if row['embedding'] is not None and not any(np.isnan(row['embedding']))]

#             # Add nodes and edges
#             for i, row in enumerate(valid_data):
#                 shloka_metadata = {
#                     "shloka": row.get("Shloka Content", "N/A"),
#                     "hindi_translation": row.get("Hindi Translation", "N/A"),
#                     "english_translation": row.get("English Translation", "N/A"),
#                     "chapter": row.get("Chapter", "N/A"),
#                     "language": row.get("Language", "N/A"),
#                     "shloka_number": row.get("Shloka", "N/A"),
#                 }

#                 # Add node with metadata
#                 graph.add_node(
#                     row['content'],
#                     embedding=row['embedding'],
#                     details=shloka_metadata
#                 )

                
#                 # Add edges based on similarity
#                 for j, other_row in enumerate(valid_data):
#                     if i < j:  # Avoid duplicate pairs
#                         sim = cosine_similarity(
#                             [row['embedding']], [other_row['embedding']]
#                         )[0][0]
#                         if sim > similarity_threshold:
#                             graph.add_edge(row['content'], other_row['content'], weight=sim)

#     # Save the graph using pickle
#     with open(output_file, 'wb') as f:
#         pickle.dump(graph, f)
#     print(f"Graph saved to {output_file}.")

def build_graph(embedding_dir, output_file, similarity_threshold=0.8):
    """
    Build a graph from embeddings and save it.
    """
    graph = nx.Graph()

    for embedding_file in os.listdir(embedding_dir):
        if embedding_file.endswith('.json'):
            with open(os.path.join(embedding_dir, embedding_file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            valid_data = [row for row in data if row['embedding'] is not None and not any(np.isnan(row['embedding']))]

            for i, row in enumerate(valid_data):
                graph.add_node(
                    row['content'], 
                    embedding=row['embedding'],
                    details={col: row[col] for col in row if col != 'embedding'}
                )

                for j, other_row in enumerate(valid_data):
                    if i < j:
                        sim = cosine_similarity([row['embedding']], [other_row['embedding']])[0][0]
                        if sim > similarity_threshold:
                            graph.add_edge(row['content'], other_row['content'], weight=sim)

    with open(output_file, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {output_file}.")
