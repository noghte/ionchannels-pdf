## PubMed RAG for Ion Channels

Helpers:
- `excel2csv.py` to convert `human_IC_annotation_details.xlsx` file to `human_IC_annotation.csv` and `human_IC_annotation_sample.csv`. 
- `extract_alternative_names.py` adds a column to the CSV files.

Steps:
1. Create Qdrant collections by running `create_chunk_vectors_qdrant.py` script. It will create a vector database in the `./qdrant` directory.
2. Run `query_rag_langchain.py` script to run queries by iterating on the `human_IC_annotation.csv` or `human_IC_annotation_sample.csv` file.

## Query examples
Is there any evidence that `anion` is the ion selectivity of the `Golgi pH regulator A(symbol: GPR89A (GPHRA, GPR89, SH120))` ion channel?

Is `Golgi pH regulator A(symbol: GPR89A (GPHRA, GPR89, SH120))` ion channel `anion` selective? 
- If true, what is the nature of the ion? (could use the prior knowledge)