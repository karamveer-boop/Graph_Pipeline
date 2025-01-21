import pickle
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import generate_embedding


def query_graph(graph_file, query, top_k=5):
    """
    Query the graph to retrieve the most similar nodes and their metadata.
    """
    # Load the graph using pickle
    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    
    # Generate query embedding
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return []
    
    # Compute similarity with all nodes
    similarities = []
    for node, data in graph.nodes(data=True):
        sim = cosine_similarity(
            [query_embedding], [data['embedding']]
        )[0][0]
        similarities.append((node, sim))
    
    # Get top-k similar nodes
    top_nodes = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Retrieve metadata (e.g., Shloka and translations) from the graph
    results = []
    for node, similarity in top_nodes:
        node_data = {
            "content": node,
            "similarity": similarity,
            "details": graph.nodes[node]['details']  # Correctly retrieve metadata
        }
        results.append(node_data)

    
    return results
