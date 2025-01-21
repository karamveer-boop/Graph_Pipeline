from .chunking import chunk_dataset
from .embeddings import process_chunk_embeddings
from .graph import build_graph
from .query import query_graph
import openai
import gradio as gr

def generate_response(graph_file, query):
    """
    Generate a response using the graph and GPT-4O.
    """
    # Retrieve contexts and metadata from the graph
    retrieved_contexts = query_graph(graph_file, query, top_k=10)
    if not retrieved_contexts:
        return "No relevant context found."
    
    # Prepare context details with metadata
    context_details = "\n\n".join(
        f"Language: {context['details'].get('language', 'N/A')}\n"
        f"Chapter: {context['details'].get('chapter', 'N/A')}\n"
        f"Shloka Number: {context['details'].get('shloka_number', 'N/A')}\n"
        f"Shloka Content: {context['details'].get('shloka', 'N/A').encode('utf-8').decode('utf-8', 'ignore')}\n"
        f"Hindi Translation: {context['details'].get('hindi_translation', 'N/A').encode('utf-8').decode('utf-8', 'ignore')}\n"
        f"English Translation: {context['details'].get('english_translation', 'N/A').encode('utf-8').decode('utf-8', 'ignore')}\n"
        f"Similarity: {context['similarity']:.2f}\n"
        for context in retrieved_contexts
    )
    
    # Formulate the prompt
    context = "\n".join([context['content'] for context in retrieved_contexts])
    prompt = (
        f"You are a highly knowledgeable assistant skilled in interpreting and providing detailed explanations of ancient texts.\n"
        f"Use the following context to answer the question in a detailed and contextual manner and behave like the knowledge giver of the write of the book context supplied:\n\nContext:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    
    # Use GPT-4O API for response generation
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,  # Increase token limit for detailed answers
        temperature=0.7,  # Adjust temperature for more nuanced responses
        top_p=0.9  # Fine-tune for relevance
    )
    model_response = response['choices'][0]['message']['content']
    
    # Combine the context details with the model response
    return f"{context_details}\n\nModel's Answer:\n{model_response}"

def initialize_and_run():
    """
    Initialize the data processing steps (chunking, embeddings, and graph building).
    """
    INPUT_FILE = "data/raw/final_sheet_1.csv"
    CHUNK_DIR = "data/chunks"
    EMBEDDING_DIR = "data/embeddings"
    GRAPH_FILE = "data/graphs/shloka_graph.pickle"

    # Step 1: Chunk the dataset
    chunk_dataset(INPUT_FILE, CHUNK_DIR, chunk_size=100)

    # # Step 2: Generate embeddings for each chunk
    # process_chunk_embeddings(CHUNK_DIR, EMBEDDING_DIR)

    # # Step 3: Build the graph
    # build_graph(EMBEDDING_DIR, GRAPH_FILE, similarity_threshold=0.8)
    
    return GRAPH_FILE


# Initialize the graph before launching the app
GRAPH_FILE = initialize_and_run()

# Define the Gradio interface
def query_interface(user_query):
    """
    Handle user query and return the detailed response.
    """
    return generate_response(GRAPH_FILE, user_query)  #

app = gr.Interface(
    fn=query_interface,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.Textbox(label="Response"),
    title="Shloka Graph RAG Assistant",
    description="This application processes shlokas, builds a graph, and provides answers to queries using GPT-3.5-turbo."
)

if __name__ == "__main__":
    app.launch(share=True)
