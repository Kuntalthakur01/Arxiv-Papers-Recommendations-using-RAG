import argparse
from typing import cast

from llama_index.core import Settings
from llama_index.core.evaluation.multi_modal import MultiModalRelevancyEvaluator
from llama_index.core.indices import MultiModalVectorStoreIndex, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from settings import EMBEDDING_MODEL, INDEX_ID, OPENAI_API_KEY
from stores import init_storage_context


# Evaluate retrieval using LLM as judge for retrieved documents
def main(args):
    Settings.embed_model = EMBEDDING_MODEL
    storage_context = init_storage_context()
    index = cast(
        MultiModalVectorStoreIndex,
        load_index_from_storage(storage_context=storage_context, index_id=INDEX_ID),
    )
    relevancy_judge = MultiModalRelevancyEvaluator(
        multi_modal_llm=OpenAIMultiModal(
            model="gpt-4o",
            max_new_tokens=300,
        )
    )
    retriever = index.as_retriever()
    retrieved_results = retriever.retrieve(args.query)
    contexts = [node.text for node in retrieved_results if type(node) != ImageDocument]
    image_paths = [
        node.image_path for node in retrieved_results if type(node) == ImageDocument
    ]

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
    response = index.as_query_engine(
        llm=llm, text_qa_template=qa_tmpl, node_postprocessors=[]
    ).query(args.query)
    print(image_paths)
    relevancy_eval = relevancy_judge.evaluate(
        query=args.query,
        response=response,
        contexts=contexts,
        image_paths=image_paths,
    )
    print(relevancy_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval using dataset")
    parser.add_argument("--query", required=True, help="Query text")
    args = parser.parse_args()
    main(args)
