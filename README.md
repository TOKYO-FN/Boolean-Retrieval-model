# Boolean Retrieval Model with GUI

This project implements a Boolean Retrieval Model with a graphical user interface (GUI) built using Tkinter. The system allows users to:

1. Load text documents from the current directory.
2. Build an inverted index for efficient document retrieval.
3. Perform Boolean queries like `term1 AND term2`, `NOT term3`, or `term4 OR term5`.

## Features

- **Inverted Index Construction**: Efficiently indexes terms across documents to enable fast retrieval.
- **Text Preprocessing**: Includes tokenization, stopword removal, and stemming using NLTK.
- **Boolean Query Support**: Supports logical operators such as `AND`, `OR`, and `NOT` for complex queries.
- **GUI Interface**: Provides an intuitive interface for loading documents, entering queries, and displaying results.

## Prerequisites

- Python 3.6+
- NLTK library (with the `stopwords` corpus downloaded)
- Tkinter (comes pre-installed with Python)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TOKYO-FN/Boolean-Retrieval-model.git
   cd Boolean-Retrieval-model
   ```

2. Download NLTK stopwords if not already installed:

   ```python
   import nltk
   nltk.download("stopwords")
   ```

## Usage

1. Place your `.txt` documents in the same directory as the script.
2. Run the application:

   ```bash
   python main_tkinter.py
   ```

3. Use the GUI to:
   - **Load Documents**: Click the "Load Documents" button to index all `.txt` files in the directory.
   - **Enter Query**: Input your Boolean query (e.g., `image AND recognition`) in the query box.
   - **Search**: Click "Search" to retrieve matching documents.

4. The matching document names will be displayed in the results section.

## Screenshots

![[Pasted image 20241221195257.png]]

## Project Structure

```
boolean-retrieval-gui/
├── main_tkinter.py                # Main script with GUI and logic
├── README.md              # Project documentation
└── *.txt                  # Place your text files here
```
