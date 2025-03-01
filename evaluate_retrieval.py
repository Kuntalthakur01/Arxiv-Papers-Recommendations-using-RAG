import argparse
from typing import cast

from llama_index.core.indices import MultiModalVectorStoreIndex, load_index_from_storage

from settings import INDEX_ID
from stores import init_storage_context


# Make changes to the following function for integration with the project
def create_index():
    storage_context = init_storage_context()
    index = cast(
        MultiModalVectorStoreIndex,
        load_index_from_storage(storage_context=storage_context, index_id=INDEX_ID),
    )
    print("Sample index created.")
    return index


# Evaluate text retrieval only
def evaluate_retrieval(query, top_k, index):
    print(f"Retrieving top {top_k} results for query: '{query}'")
    retriever = index.as_retriever()
    retrieved_results = retriever.retrieve(query)

    print("\nRetrieved Results:")
    for i, result in enumerate(retrieved_results):
        print(f"  Result {i+1}: {result.text[:100]}...")

    evaluator = RetrieverEvaluator.from_metric_names(
        metric_names=["mrr", "context_relevancy", "recall"], retriever=retriever
    )
    evaluation_result = evaluator.evaluate(
        query=query, expected_ids=[], retrieved_results=retrieved_results
    )

    print("\nEvaluation Metrics:")
    print(
        f"  Mean Reciprocal Rank (MRR): {evaluation_result.metric_dict['mrr'].value:.4f}"
    )
    print(
        f"  Context Relevance: {evaluation_result.metric_dict['context_relevancy'].value:.4f}"
    )
    print(f"  Recall: {evaluation_result.metric_dict['recall'].value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval using dataset")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top results to retrieve"
    )
    args = parser.parse_args()
    index = create_sample_index()
    evaluate_retrieval(query=args.query, top_k=args.top_k, index=index)
