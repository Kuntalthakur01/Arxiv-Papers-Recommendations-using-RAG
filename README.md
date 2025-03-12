# Arxiv-Papers-Recommendations-using-RAG


## Env Setup

- Run the following commands to create python env with all dependencies

```
conda create -n <env-name>
conda activate <env-name>
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
conda install conda-forge::python-dotenv
pip install llama-index
pip install llama-index-embeddings-clip
pip install git+https://github.com/openai/CLIP.git
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-faiss
pip install llama-index-storage-docstore-dynamodb
pip install llama-index-storage-index-store-dynamodb-store
```

- Edit the [.env](./.env) file with the requred secrets

- Ensure the required tables are created in DynamoDB. You can find the table names in the [Settings](./settings.py) file. The tables should have a hash key and partition key of type String

## Running

### Scraping and Preprocessing

### The scraping and preprocessing steps involve extracting and preparing the data for further processing.

## Scraping Data

#### 1. Scraping PDF Papers:

• Use the scrape_arxiv-papers.ipynb script to scrape research papers from arXiv.

• The script downloads PDF files for further processing.

## Preprocessing

#### 2. Scraping Latex files & Equation Extraction:

• Scraping research paper in Latex format using the LatexScrape.py script.

• Use the equation.py script to extract mathematical equations from the LaTeX files and give a JSON file for each paper.

#### 3. JSON Conversion and Metadata Creation:

• Use Extraction.ipynb to extract text and tables from the papers for structured representation and save them in JSON format.

• Extract images from the papers and save them in a separate folder for each arxiv paper.

• Use the extracted JSON files and image files to generate metadata for each paper.

• Generate a structured directory for all the extracted data for efficient retrieval and processing.

For Eg:

<img width="921" alt="Screenshot 2024-12-03 at 11 50 16 PM" src="https://github.com/user-attachments/assets/3ee78c43-746b-415c-bbfb-eb180fee5081">

### Creating the RAG index

- To populate the docstore, vector store and index run the following

```
python populate_index.py --dataset-path=<path-to-dataset-folder>
```

- The dataset path is the path to the folder created by the preprocessing script

### Query LLM

- Run the following command to query the LLM with RAG

```
python query_llm.py --query="<query-text>"
```
