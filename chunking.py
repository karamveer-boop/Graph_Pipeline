import os
import pandas as pd
from .utils import detect_encoding

import os
import pandas as pd
from .utils import detect_encoding

def chunk_dataset(input_file, output_dir, chunk_size=100):
    """
    Splits the dataset into smaller chunks for easier processing and ensures proper encoding handling.
    """
    os.makedirs(output_dir, exist_ok=True)
    encoding = detect_encoding(input_file)
    print(f"Detected file encoding: {encoding}")

    # If encoding is IBM866, convert to UTF-8
    if encoding == 'IBM866':
        data = pd.read_csv(input_file, encoding=encoding)
        # Convert columns to utf-8
        for column in data.select_dtypes(include=['object']).columns:
            data[column] = data[column].apply(lambda x: x.encode('IBM866').decode('utf-8', 'ignore') if isinstance(x, str) else x)
    else:
        data = pd.read_csv(input_file, encoding=encoding)
    
    # Split the dataset into chunks and save them
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        chunk.to_csv(os.path.join(output_dir, f"chunk_{i // chunk_size}.csv"), index=False, encoding='utf-8')
    print(f"Dataset split into chunks and saved to {output_dir}.")
