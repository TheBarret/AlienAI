import os
import re
import logging
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Purpose of this script:
# - Load the dataset
# - Remove special characters
# - Remove numbers
# - Remove extra whitespaces
# - Convert to lowercase
# - Remove stop words
# - Stemming

# Configure logging
logging.basicConfig(level=logging.INFO)

# Dataset loading
def pp_load(file_path):
    logging.info(f"[preprocessor] Opening '{file_path}'...")
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            dataset = pp_clean(file.read())
        return dataset
    logging.info("[preprocessor] file not found")

# Dataset cleaning
def pp_clean(text):
    logging.info(f"[preprocessor] Removing special characters")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    logging.info(f"[preprocessor] Removing numbers")
    text = re.sub(r'\d+', '', text)
    logging.info(f"[preprocessor] Removing extra whitespaces")
    text = ' '.join(text.split())
    logging.info(f"[preprocessor] Converting to lowercase")
    text = text.lower()
    # Tokenize the text before removing stop words
    tokens = text.split()
    logging.info(f"[preprocessor] Removing stop words")
    words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in words]
    logging.info(f"[preprocessor] Stemming")
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text
