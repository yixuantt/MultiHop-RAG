# 💡 MultiHop-RAG
A Dataset for **Evaluating Retrieval-Augmented Generation Across Documents**  

   
# 🚀 Overview
**MultiHop-RAG**: a QA dataset to evaluate retrieval and reasoning across documents with metadata in the RAG pipelines. It contains 2556 queries, with evidence for each query distributed across 2 to 4 documents. The queries also involve document metadata, reflecting complex scenarios commonly found in real-world RAG applications.  

📄 Paper Link **(Accepted by COLM 2024)**: [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/pdf/2401.15391.pdf)  
🤗 [Hugging Face dataloader](https://huggingface.co/datasets/yixuantt/MultiHopRAG)

![rag.png](resource/rag.png)

# Simple Use Case

**1. For Retrieval**

Please try '**simple_retrieval.py**,' a sample use case demonstrating retrieval using this dataset. 
```
pip install llama-index==0.9.40
```
```shell
# test simple retrieval and save results
python simple_retrieval.py --retriever BAAI/llm-embedder

# test simple retrieval with rerank and save results
python simple_retrieval.py --retriever BAAI/llm-embedder --rerank
```

**2. For QA**

Please try '**qa_llama.py**,' a sample use case demonstrating query and answer with llama using this dataset. 

```
python qa_llama.py
```
# Evaluation

**1. For Retrieval**: 'retrieval_evaluate.py' 

**2. For QA**: 'qa_evaluate.py' 
```
python retrieval_evaluate.py --file {saved_file_path}
```

# Citation
```
@misc{tang2024multihoprag,
      title={MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries}, 
      author={Yixuan Tang and Yi Yang},
      year={2024},
      eprint={2401.15391},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
# License
MultiHop-RAG is licensed under [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)
