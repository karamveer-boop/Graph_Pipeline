import chardet
import os
import pandas as pd
# ----------------------------------------------
# STEP 1: Detect File Encoding
# ----------------------------------------------
def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']


# ----------------------------------------------
# STEP 2: Truncate Text
# ----------------------------------------------
def truncate_text(text, max_chars=2000):
    """
    Truncate text to ensure it fits within the model's context length.
    """
    if len(text) > max_chars:
        return text[:max_chars]
    return text

def chunk_dataset(input_file, output_dir, chunk_size=100):
    """
    Splits the dataset into smaller chunks for easier processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect encoding
    encoding = detect_encoding(input_file)
    print(f"Detected file encoding: {encoding}")
    
    # Read the CSV with the detected encoding
    data = pd.read_csv(input_file, encoding=encoding)
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        chunk.to_csv(os.path.join(output_dir, f"chunk_{i // chunk_size}.csv"), index=False)
    print(f"Dataset split into chunks and saved to {output_dir}.")
