import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
import re
import nltk

class BooleanRetrievalModel:
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = [self.stemmer.stem(word) for word in text.split() if word not in self.stop_words]
        return tokens

    def build_index(self, documents):
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

      for i, token in enumerate(tokens):
        if token in operators:
          current_op = token
        else:
				# Preprocess the token to match the document preprocessing
          preprocessed_token = self.preprocess_text(token)
          token_docs = self.inverted_index.get(preprocessed_token[0], set()) if preprocessed_token else set()

					# Handle "AND NOT" explicitly
          if i > 1 and tokens[i - 1] == "not" and tokens[i - 2] == "and":
              if result_stack:
                  prev_docs = result_stack.pop()
                  token_docs = prev_docs - token_docs
              current_op = None  # Reset operator

					# Handle "OR NOT" explicitly
          elif i > 1 and tokens[i - 1] == "not" and tokens[i - 2] == "or":
              if result_stack:
                  prev_docs = result_stack.pop()
                  token_docs = prev_docs | (set(documents.keys()) - token_docs)
              current_op = None  # Reset operator

					# Apply NOT operator standalone
          elif current_op == "not":
              universe = set(documents.keys())
              token_docs = universe - token_docs
              current_op = None  # Reset operator

					# Process the result stack for "AND" and "OR"
          if result_stack:
              prev_docs = result_stack.pop()
              if current_op == "and":
                  token_docs = prev_docs & token_docs
              elif current_op == "or":
                  token_docs = prev_docs | token_docs

          result_stack.append(token_docs)
          current_op = None  # Reset operator

      return result_stack.pop() if result_stack else set()


def load_documents_from_txt_files():
    documents = {}
    for file_name in os.listdir():
        if file_name.endswith(".txt"):
            with open(file_name, "r", encoding="utf-8") as file:
                documents[file_name] = file.read()
    return documents

class BooleanRetrievalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Boolean Retrieval Model")

        self.brm = BooleanRetrievalModel()
        self.documents = {}

        self.label = tk.Label(root, text="Boolean Retrieval System", font=("Arial", 16))
        self.label.pack(pady=10)

        self.load_button = tk.Button(root, text="Load Documents", command=self.load_documents)
        self.load_button.pack(pady=5)

        self.query_label = tk.Label(root, text="Enter Query:")
        self.query_label.pack(pady=5)

        self.query_entry = tk.Entry(root, width=50)
        self.query_entry.pack(pady=5)

        self.search_button = tk.Button(root, text="Search", command=self.search_query)
        self.search_button.pack(pady=5)

        self.result_text = tk.Text(root, height=20, width=80, wrap=tk.WORD)
        self.result_text.pack(pady=10)

    def load_documents(self):
        self.documents = load_documents_from_txt_files()
        if not self.documents:
            messagebox.showerror("Error", "No .txt files found in the current directory!")
        else:
            self.brm.build_index(self.documents)
            messagebox.showinfo("Success", f"Loaded {len(self.documents)} documents and built the index.")

    def search_query(self):
        query = self.query_entry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a query.")
            return

        results = self.brm.boolean_query(query, self.documents)
        self.result_text.delete(1.0, tk.END)

        if results:
            self.result_text.insert(tk.END, "Matching Documents:\n\n")
            for doc in results:
                self.result_text.insert(tk.END, f"{doc}:\n")
                self.result_text.insert(tk.END, f"{self.documents[doc]}\n\n{'-'*40}\n\n")
        else:
            self.result_text.insert(tk.END, "No matching documents found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BooleanRetrievalApp(root)
    root.mainloop()
