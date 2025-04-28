"""
name: Process PR Data
on:
  pull_request:
    types: [opened, edited, reopened, synchronize]

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      # Step 1: Checkout code (required for accessing requirements.txt and my-script.py)
      - uses: actions/checkout@v4

      # Step 2: Set Up Python
      - name: Set Up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      # Step 3: Install dependencies
      - name: Install requirements
        run: |
          pip install -r requirements.txt

      # Step 4: Get PR Description
      - name: Extract PR Description
        id: pr-description
        run: |
          echo "DESCRIPTION<<EOF" >> $GITHUB_OUTPUT
          echo "${{ github.event.pull_request.body || '[No description]' }}" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      # Step 5: Fetch URL Contents
      - name: Fetch URL Data
        id: fetch-url
        run: |
          curl -sSL "https://my-url.com" -o fetched-content.txt
          echo "CONTENT=$(cat fetched-content.txt)" >> $GITHUB_OUTPUT

      # Step 6: Run Script
      - name: Execute Analysis Script
        run: |
          python my-script.py \
            "${{ steps.pr-description.outputs.DESCRIPTION }}" \
            "${{ steps.fetch-url.outputs.CONTENT }}"

"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def initialize_vectore_store(filename):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    db_location = "./db"
    vector_store = Chroma(
        collection_name="documentation",
        embedding_function=embeddings,
        persist_directory=db_location,
    )
    process_document(filename, vector_store)
    return vector_store


def add_chunks_to_vector_store(chunks, file, vector_store):
    for i, chunk in enumerate(chunks):
        chunk_id = f"{file}_chunk_{i}"

        chunk.metadata.update(
            {"source": file, "chunk_number": str(i)}
        )

        vector_store.add_documents(documents=[chunk], ids=[chunk_id])
    print(f"Completed: Added {str(i+1)} chunks from {file} to vector store")


def process_document(filename, vector_store):
    loader = BSHTMLLoader(filename)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(document)
    add_chunks_to_vector_store(chunks, filename, vector_store)


def get_retriever(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever


def main(update_message, existing_doc):
    print("in main")

    model = OllamaLLM(model="llama3.2")

    template = """
    Here is the existing documentation for a code repository: {documentation}

    Here is the description of an update the user want to make to the code repository: {update_message}

    Should the user update the existing documentation to match the changes they are making? If so, specify what changes they should make. If not, give a one sentence explanation of why.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model
    seen = {}
    vector_store = initialize_vectore_store(existing_doc)
    print("vector store initialized")

    retriever = get_retriever(vector_store)
    documentation = retriever.invoke(existing_doc)
    result = chain.invoke({"documentation": documentation, "update_message": update_message})
    print(result)
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python llm-compare.py <update_message> <existing_doc>")
        sys.exit(1)
        
    update_message = sys.argv[1]
    existing_doc = sys.argv[2]
    main(update_message, existing_doc)




