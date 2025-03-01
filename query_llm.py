import argparse
from typing import cast

from llama_index.core import PromptTemplate, Settings
from llama_index.core.indices import MultiModalVectorStoreIndex, load_index_from_storage
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from settings import EMBEDDING_MODEL, INDEX_ID, OPENAI_API_KEY
from stores import init_storage_context


def main(args):
    Settings.embed_model = EMBEDDING_MODEL
    storage_context = init_storage_context()
    index = cast(
        MultiModalVectorStoreIndex,
        load_index_from_storage(storage_context=storage_context, index_id=INDEX_ID),
    )

    qa_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_tmpl = PromptTemplate(qa_tmpl_str)

    llm = OpenAIMultiModal(model="gpt-4o", api_key=OPENAI_API_KEY, max_new_tokens=200)
    print("OUTPUT")
    print(
        index.as_query_engine(
            llm=llm, text_qa_template=qa_tmpl, node_postprocessors=[]
        ).query(args.query)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--query", required=True, help="The query input to the LLM", type=str
    )
    args = parser.parse_args()

    main(args)
