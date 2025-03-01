# Use this script to rename metadata file from <paper-id>_metadata.json to metadata.json
# This will be needed if dataset.py fails to load the metadata file, since its name is in the wrong format


import os

folders = os.listdir("./2024_arxiv_papers_CS/")

for folder in folders:
    if os.path.exists(f"./2024_arxiv_papers_CS/{folder}/{folder}_metadata.json"):
        os.rename(
            f"./2024_arxiv_papers_CS/{folder}/{folder}_metadata.json",
            f"./2024_arxiv_papers_CS/{folder}/metadata.json",
        )
