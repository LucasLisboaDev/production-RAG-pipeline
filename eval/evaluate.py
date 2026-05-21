"""
RAGAs Evaluation Pipeline (Week 4)
=====================================
Runs the golden dataset through the full RAG pipeline and
measures faithfulness + context recall using RAGAs.

Run locally:
    pytest eval/evaluate.py -v

The CI pipeline runs this automatically on every PR.
"""

import json
import pytest
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, context_recall, answer_relevancy

from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import Reranker
from generation.prompt_builder import build_prompt
from generation.llm_client import call_llm

GOLDEN_PATH = Path("eval/golden_dataset.json")

# Thresholds — CI fails if scores drop below these
FAITHFULNESS_THRESHOLD   = 0.75
CONTEXT_RECALL_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70


@pytest.fixture(scope="module")
def pipeline():
    return {
        "retriever": HybridRetriever(top_k=20),
        "reranker":  Reranker(),
    }


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ragas_dataset(pipeline, golden):
    retriever = pipeline["retriever"]
    reranker  = pipeline["reranker"]

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        question = item["question"]
        candidates = retriever.retrieve(question)
        top_chunks = reranker.rerank(question, candidates, top_k=5)
        messages   = build_prompt(question, top_chunks)
        answer     = call_llm(messages)

        questions.append(question)
        answers.append(answer)
        contexts.append([c["text"] for c in top_chunks])
        ground_truths.append(item["expected_answer"])

    return Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })


def test_faithfulness(ragas_dataset):
    result = evaluate(ragas_dataset, metrics=[faithfulness])
    score = result["faithfulness"]
    print(f"\n  Faithfulness: {score:.3f} (threshold: {FAITHFULNESS_THRESHOLD})")
    assert score >= FAITHFULNESS_THRESHOLD, (
        f"Faithfulness {score:.3f} below threshold {FAITHFULNESS_THRESHOLD}"
    )


def test_context_recall(ragas_dataset):
    result = evaluate(ragas_dataset, metrics=[context_recall])
    score = result["context_recall"]
    print(f"\n  Context Recall: {score:.3f} (threshold: {CONTEXT_RECALL_THRESHOLD})")
    assert score >= CONTEXT_RECALL_THRESHOLD, (
        f"Context recall {score:.3f} below threshold {CONTEXT_RECALL_THRESHOLD}"
    )


def test_answer_relevancy(ragas_dataset):
    result = evaluate(ragas_dataset, metrics=[answer_relevancy])
    score = result["answer_relevancy"]
    print(f"\n  Answer Relevancy: {score:.3f} (threshold: {ANSWER_RELEVANCY_THRESHOLD})")
    assert score >= ANSWER_RELEVANCY_THRESHOLD, (
        f"Answer relevancy {score:.3f} below threshold {ANSWER_RELEVANCY_THRESHOLD}"
    )
