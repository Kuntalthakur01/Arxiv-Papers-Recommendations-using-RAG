import argparse
import os
import shutil
from typing import cast

from llama_index.core import Document, Settings, load_index_from_storage
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.ingestion.pipeline import run_transformations
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import ImageDocument, Node

from dataset import ArxivPaperDataset
from settings import DATASET_PATH, EMBEDDING_MODEL, INDEX_ID, PERSIST_PATH
from stores import init_storage_context


def main(args):
    Settings.embed_model = EMBEDDING_MODEL

    arxiv_papers = ArxivPaperDataset(DATASET_PATH)
    storage_context = init_storage_context()
    vector_index: MultiModalVectorStoreIndex

    try:
        print("Loading pre-existing index")
        vector_index = cast(
            MultiModalVectorStoreIndex,
            load_index_from_storage(storage_context=storage_context, index_id=INDEX_ID),
        )
    except:
        print("Pre existing index not found. Creating new one")
        vector_index = MultiModalVectorStoreIndex.from_documents(
            [], storage_context=storage_context
        )
        vector_index.set_index_id(INDEX_ID)
    print(type(vector_index))
    print("Done")

    text_transformations = [
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=EMBEDDING_MODEL,
        )
    ]

    print("Indexing Papers")

    indexed_count = 0
    for ind, paper in enumerate(arxiv_papers):
        # TODO: Remove this. Just for testing
        if paper["images"] == []:
            continue
        paper["equations"] = []

        try:
            text_documents = [
                Document(text=paper["sections"][section_name])
                for section_name in paper["sections"].keys()
                if section_name != "Tables"
            ]
            text_nodes = run_transformations(text_documents, text_transformations)  # type: ignore
            equation_nodes = [
                Document(text=equation["latex"]) for equation in paper["equations"]
            ]
            image_documents = [
                ImageDocument(image_path=image["path"]) for image in paper["images"]
            ]

            vector_index.insert_nodes(image_documents + text_nodes + equation_nodes)  # type: ignore

            indexed_count += 1
            print(f"Indexed Paper at: {paper['paper_path']}")
        except Exception as e:
            print(e)
            pass

        if args.limit and indexed_count >= args.limit:
            break

    if os.path.exists(PERSIST_PATH):
        shutil.rmtree(PERSIST_PATH)

    storage_context.persist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="The path to dataset containing preprocessed PDF data",
        type=str,
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of files added to index. Mainly for testing purpose",
    )
    args = parser.parse_args()

    main(args)
