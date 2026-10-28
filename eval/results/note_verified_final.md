# Literature watch note: evaluation of RAG systems

_Every claim below was checked against the title and abstract of the paper it cites._

## RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework [P1]
- RAGEval generates scenario‑specific evaluation data by producing high‑quality documents, questions, answers, and references through a schema‑based pipeline. [P1]
- RAGEval introduces three novel metrics: Completeness, Hallucination, and Irrelevance. [P1]
- Experimental results show RAGEval outperforms zero‑shot and one‑shot methods in clarity, safety, conformity, and richness of generated samples. [P1]
- Experimental results show RAGEval outperforms zero‑shot and one‑shot methods in clarity, safety, conformity, and richness of generated samples. [P1]
- RAGEval generates scenario‑specific evaluation data by producing high‑quality documents, questions, answers, and references through a schema‑based pipeline. [P1]

## Deepchecks: Evaluating Retrieval-Augmented Generation (RAG) [P2]
- Deepchecks is a comprehensive framework for evaluating Retrieval‑Augmented Generation applications. [P2]
- Deepchecks addresses evaluation through a multi‑faceted approach, root cause analysis, and production monitoring. [P2]
- Deepchecks provides a robust foundation for assessing reliability, relevance, and user satisfaction in RAG systems. [P2]

## EvaRAG: Evaluating Advanced RAG Techniques With Indexing and Distance Metrics [P3]
- EvaRAG evaluates RAG configurations by systematically varying retrievers, indexing methods, similarity metrics, and rerankers across small, medium, and large datasets. [P3]
- The HNSW–IP–Fusion–minilm configuration achieves Coverage Retrieval 0.942, Correctness 0.909, and Faithfulness 0.970. [P3]
- The IVF–L2–Hierarchical configuration demonstrates the lowest latency of 1.736 ns and lowest cost. [P3]
- HNSW–IP–Fusion–minilm achieves Coverage Retrieval of 0.942, Correctness of 0.909, and Faithfulness of 0.970. [P3]
- IVF–L2–Hierarchical demonstrates the lowest latency of 1.736 ns and cost, making it suitable for real-time deployments. [P3]
- The HNSW–IP–Fusion–minilm configuration achieves Coverage Retrieval 0.942, Correctness 0.909, and Faithfulness 0.970. [P3]
- The IVF–L2–Hierarchical configuration demonstrates the lowest latency of 1.736 ns and lowest cost. [P3]

## RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation [P4]
- RAGChecker is a fine‑grained evaluation framework that incorporates diagnostic metrics for both retrieval and generation modules. [P4]
- Meta evaluation shows RAGChecker has significantly better correlations with human judgments than other evaluation metrics. [P4]
- Using RAGChecker, 8 RAG systems were evaluated, revealing insightful patterns and trade‑offs in design choices. [P4]

## RAGe: A Retrieval-Augmented Generation Evaluation Framework [P5]
- RAGe is a modular framework for benchmarking RAG applications that focuses on resource telemetry and component recommendation. [P5]
- RAGe correlates retrieval and generation quality with underlying hardware constraints to guide domain‑specific RAG setups. [P5]
- RAGe supports rapid prototyping even on consumer‑grade hardware. [P5]

## EncouRAGe: Evaluating RAG Local, Fast, and Reliable [P6]
- EncouRAGe comprises five modular components: Type Manifest, RAG Factory, Inference, Vector Store, and Metrics. [P6]
- Evaluation across 25k QA pairs and 51k documents shows RAG still underperforms compared to the Oracle Context. [P6]
- Hybrid BM25 consistently achieves the best results across all four datasets, while reranking yields only marginal improvements with higher latency. [P6]
- Hybrid BM25 consistently achieves the best results across all four datasets, while reranking yields only marginal performance improvements accompanied by higher response latency. [P6]

## Revisiting the Solution of Meta KDD Cup 2024: CRAG [P7]
- CRAG benchmark addresses limitations of existing QA benchmarks for evaluating diverse and dynamic challenges faced by RAG systems. [P7]
- We propose a routing-based domain and dynamic adaptive RAG pipeline. [P7]
- Our method achieved superior performance on CRAG and ranked 2nd for Task 2&3 on the final competition leaderboard. [P7]

## RAG Playground: A Framework for Systematic Evaluation of Retrieval Strategies and Prompt Engineering in RAG Systems [P8]
- RAG Playground implements naive vector search, reranking, and hybrid vector‑keyword search combined with ReAct agents. [P8]
- Experiments show hybrid search methods and structured self‑evaluation prompting achieve up to 72.7% pass rate on the multi‑metric evaluation framework. [P8]
- Custom‑prompted agents consistently improve retrieval accuracy and response quality. [P8]
- Our experiments demonstrate that hybrid search methods and structured self-evaluation prompting achieve up to 72.7% pass rate on the multi-metric evaluation framework. [P8]
- Experiments show hybrid search methods and structured self‑evaluation prompting achieve up to 72.7% pass rate on the multi‑metric evaluation framework. [P8]
- Custom‑prompted agents show consistent improvements in retrieval accuracy and response quality. [P8]

## THELMA: Task Based Holistic Evaluation of Large Language Model Applications-RAG Question Answering [P9]
- THELMA is a reference‑free framework with six interdependent metrics for holistic evaluation of RAG QA applications. [P9]
- THELMA enables developers to evaluate, monitor, and improve end‑to‑end RAG QA pipelines without labeled sources or reference responses. [P9]
- Findings on the interplay of THELMA metrics can identify the specific RAG component needing improvement. [P9]
- Findings on the interplay of THELMA metrics can be interpreted to identify the specific RAG component needing improvement in QA applications. [P9]

## Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation [P10]
- Semantic stratification grounds retrieval evaluation in corpus structure by organizing documents into entity‑based clusters and generating queries for missing strata. [P10]
- Semantic stratification provides formal semantic coverage guarantees across retrieval regimes. [P10]
- Experiments show stratified evaluation yields more stable and transparent assessments than aggregate metrics. [P10]
- Semantic stratification grounds retrieval evaluation in corpus structure by organizing documents into entity‑based clusters and systematically generating queries for missing strata. [P10]

## References

- [P1] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework (2024) - http://arxiv.org/abs/2408.01262v5
- [P2] Deepchecks: Evaluating Retrieval-Augmented Generation (RAG) (2026) - https://doi.org/10.48550/arxiv.2605.14488
- [P3] EvaRAG: Evaluating Advanced RAG Techniques With Indexing and Distance Metrics (2025) - https://doi.org/10.1109/access.2025.3646665
- [P4] RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation (2024) - http://arxiv.org/abs/2408.08067v2
- [P5] RAGe: A Retrieval-Augmented Generation Evaluation Framework (2026) - https://doi.org/10.48550/arxiv.2605.27445
- [P6] EncouRAGe: Evaluating RAG Local, Fast, and Reliable (2025) - http://arxiv.org/abs/2511.04696v1
- [P7] Revisiting the Solution of Meta KDD Cup 2024: CRAG (2024) - http://arxiv.org/abs/2409.15337v1
- [P8] RAG Playground: A Framework for Systematic Evaluation of Retrieval Strategies and Prompt Engineering in RAG Systems (2024) - http://arxiv.org/abs/2412.12322v1
- [P9] THELMA: Task Based Holistic Evaluation of Large Language Model Applications-RAG Question Answering (2025) - http://arxiv.org/abs/2505.11626v2
- [P10] Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation (2026) - http://arxiv.org/abs/2604.20763v1
