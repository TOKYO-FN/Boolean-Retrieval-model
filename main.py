import os
import re
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

class BooleanRetrievalModel:
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess_text(self, text):
        """
        Preprocess the text: lowercase, remove punctuation, tokenize, remove stopwords, and stem.
        
        :param text: The raw text input.
        :return: A list of preprocessed terms.
        """
        # Lowercase the text
        text = text.lower()
        # Remove punctuation and non-alphanumeric characters
        text = re.sub(r"[^a-z0-9\s]", "", text)
        # Tokenize, remove stopwords, and apply stemming
        tokens = [self.stemmer.stem(word) for word in text.split() if word not in self.stop_words]
        return tokens

    def build_index(self, documents):
        """
        Build the inverted index from a collection of documents.
        
        :param documents: A dictionary where keys are document IDs and values are document texts.
        """
        for doc_id, text in documents.items():
            terms = self.preprocess_text(text)
            for term in terms:
                self.inverted_index[term].add(doc_id)

    def boolean_query(self, query, documents):
        """
        Perform a Boolean query on the inverted index.
        
        :param query: A string Boolean query (e.g., "term1 AND term2 OR term3").
        :return: A set of document IDs matching the query.
        """
        tokens = query.lower().split()
        result_stack = []
        operators = {"and", "or", "not"}
        current_op = None

        for token in tokens:
            if token in operators:
                current_op = token
            else:
                # Preprocess the token to match the document preprocessing
                preprocessed_token = self.preprocess_text(token)
                token_docs = self.inverted_index.get(preprocessed_token[0], set()) if preprocessed_token else set()

                # Apply NOT operator
                if current_op == "not":
                    universe = set(documents.keys())
                    token_docs = universe - token_docs
                    current_op = None  # Reset operator

                # Process the result stack
                if result_stack:
                    prev_docs = result_stack.pop()
                    if current_op == "and":
                        token_docs = prev_docs & token_docs
                    elif current_op == "or":
                        token_docs = prev_docs | token_docs

                result_stack.append(token_docs)
                current_op = None  # Reset operator

        return result_stack.pop() if result_stack else set()


# Function to load .txt files
def load_documents_from_txt_files():
    """
    Load all text files in the current directory and create a dictionary
    where the key is the file name and the value is the file content.
    
    :return: A dictionary with file names as keys and file contents as values.
    """
    documents = {}
    for file_name in os.listdir():
        if file_name.endswith(".txt"):  # Check if the file is a .txt file
            with open(file_name, "r", encoding="utf-8") as file:
                documents[file_name] = file.read()
    return documents


def calculate_precision_recall(retrieved_docs, relevant_docs):
    """
    Calculate precision and recall based on retrieved and relevant documents.
    
    :param retrieved_docs: A set of document IDs retrieved by the query.
    :param relevant_docs: A set of document IDs that are relevant (ground truth).
    :return: A tuple of (precision, recall).
    """
    if not retrieved_docs:
        precision = 0
    else:
        precision = len(retrieved_docs & relevant_docs) / len(retrieved_docs)

    if not relevant_docs:
        recall = 0
    else:
        recall = len(retrieved_docs & relevant_docs) / len(relevant_docs)

    return precision, recall


# Load documents
documents = load_documents_from_txt_files()

# Initialize and build the index
brm = BooleanRetrievalModel()

# brm.preprocess_text()
brm.build_index(documents)

# Example ground truth for each query
ground_truth = {
    "image OR recognition": {"deep_learning.txt", "machine_learning.txt"},
    "hello OR left": {"try.txt", "el 4bab el 7lw.txt", "information_retrieval.txt"},
    "not neural and machine": {"machine_learning.txt"},
}

# Perform queries and calculate precision and recall
queries = [
"image OR recognition",
"hello OR left",
"not neural and machine",
]

for query in queries:
    retrieved_docs = brm.boolean_query(query, documents)
    relevant_docs = ground_truth[query]
    precision, recall = calculate_precision_recall(retrieved_docs, relevant_docs)

    print(f"Query: '{query}'")
    print(f"Retrieved: {retrieved_docs}")
    print(f"Relevant: {relevant_docs}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}\n")
