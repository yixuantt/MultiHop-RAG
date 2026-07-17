#
#
#
# import json
# from util import rm_file
# from tqdm import tqdm
# import argparse
# from copy import deepcopy
# import os
# from util import JSONReader
# import openai
# from typing import List, Dict
#
# from llama_index import (
#     ServiceContext,
#     OpenAIEmbedding,
#     PromptHelper,
#     VectorStoreIndex,
#     set_global_service_context
# )
# from llama_index.extractors import BaseExtractor
# from llama_index.ingestion import IngestionPipeline
# from llama_index.embeddings.cohereai import CohereEmbedding
# from llama_index.llms import OpenAI
# from llama_index.text_splitter import SentenceSplitter
# from llama_index.embeddings import HuggingFaceEmbedding, VoyageEmbedding, InstructorEmbedding
# from llama_index.postprocessor import FlagEmbeddingReranker
# from llama_index.schema import QueryBundle, MetadataMode
#
#
# class CustomExtractor(BaseExtractor):
#     async def aextract(self, nodes) -> List[Dict]:
#         metadata_list = [
#             {
#                 "title": (
#                     node.metadata["title"]
#                 ),
#                 "source": (
#                     node.metadata["source"]
#                 ),
#                 "published_at": (
#                     node.metadata["published_at"]
#                 )
#             }
#             for node in nodes
#         ]
#         return metadata_list
#
#
# if __name__ == '__main__':
#     openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
#     openai.base_url = "your_api_base"
#     voyage_api_key = os.environ.get("VOYAGE_API_KEY", "your_voyage_api_key")
#     cohere_api_key = os.environ.get("COHERE_API_KEY", "your_cohere_api_key")
#
#     parser = argparse.ArgumentParser(description="running script.")
#     parser.add_argument('--retriever', type=str, required=True, help='retriever name')
#     parser.add_argument('--llm', type=str, required=False, default="gpt-3.5-turbo-1106", help='LLMs')
#     parser.add_argument('--rerank', action='store_true', required=False, default=False, help='if rerank')
#     parser.add_argument('--topk', type=int, required=False, default=10, help='Top K')
#     parser.add_argument('--chunk_size', type=int, required=False, default=256, help='chunk_size')
#     parser.add_argument('--context_window', type=int, required=False, default=2048, help='context_window')
#     parser.add_argument('--num_output', type=int, required=False, default=256, help='num_output')
#
#     args = parser.parse_args()
#     os.makedirs("output", exist_ok=True)
#
#     model_name = args.retriever
#     rerank = args.rerank
#     top_k = args.topk
#     save_model_name = model_name.split('/')
#     llm = OpenAI(model=args.llm, temperature=0, max_tokens=args.context_window)
#
#     # define save file
#     if rerank:
#         save_file = f'output/{save_model_name[-1]}_rerank_retrieval_test.json'
#     else:
#         save_file = f'output/{save_model_name[-1]}_retrieval_test.json'
#     rm_file(save_file)
#     print(f'save_file:{save_file}')
#
#     if 'text' in model_name:
#         # "text-embedding-ada-002" “text-search-ada-query-001”
#         embed_model = OpenAIEmbedding(model=model_name, embed_batch_size=10)
#     elif 'Cohere' in model_name:
#         embed_model = CohereEmbedding(
#             cohere_api_key=cohere_api_key,
#             model_name="embed-english-v3.0",
#             input_type="search_query",
#         )
#     elif 'voyage-02' in model_name:
#         embed_model = VoyageEmbedding(
#             model_name='voyage-02', voyage_api_key=voyage_api_key
#         )
#     elif 'instructor' in model_name:
#         embed_model = InstructorEmbedding(model_name=model_name)
#     else:
#         embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)
#
#     # service context
#     text_splitter = SentenceSplitter(chunk_size=args.chunk_size)
#     prompt_helper = PromptHelper(
#         context_window=args.context_window,
#         num_output=args.num_output,
#         chunk_overlap_ratio=0.1,
#         chunk_size_limit=None,
#     )
#     service_context = ServiceContext.from_defaults(
#         llm=llm,
#         embed_model=embed_model,
#         text_splitter=text_splitter,
#         prompt_helper=prompt_helper,
#     )
#     set_global_service_context(service_context)
#
#     reader = JSONReader()
#     data = reader.load_data('dataset/corpus.json')
#     # print(data[0])
#
#     transformations = [text_splitter, CustomExtractor()]
#     pipeline = IngestionPipeline(transformations=transformations)
#     nodes = pipeline.run(documents=data)
#
#     print(f"Total documents: {len(data)}")
#     print(f"Total nodes/chunks: {len(nodes)}")
#
#     nodes_see = deepcopy(nodes)
#     print(
#         "LLM sees:\n",
#         (nodes_see)[0].get_content(metadata_mode=MetadataMode.LLM),
#     )
#     print('Finish Loading...')
#
#     index = VectorStoreIndex(nodes, show_progress=True)
#     print('Finish Indexing...')
#
#     with open('dataset/MultiHopRAG.json', 'r') as file:
#         query_data = json.load(file)
#
#     if rerank:
#         rerank_postprocessors = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=top_k)
#
#     # test retrieval quality
#     retrieval_save_list = []
#     print("start to retrieve...")
#     for data in tqdm(query_data):
#         query = data['query']
#         if rerank:
#             nodes_score = index.as_retriever(similarity_top_k=20).retrieve(query)
#             nodes_score = rerank_postprocessors.postprocess_nodes(
#                 nodes_score, query_bundle=QueryBundle(query_str=query)
#             )
#         else:
#             nodes_score = index.as_retriever(similarity_top_k=top_k).retrieve(query)
#
#         retrieval_list = []
#         for ns in nodes_score:
#             dic = {}
#             dic['text'] = ns.get_content(metadata_mode=MetadataMode.LLM)
#             dic['score'] = ns.get_score()
#             retrieval_list.append(dic)
#
#         save = {}
#         save['query'] = data['query']
#         save['answer'] = data['answer']
#         save['question_type'] = data['question_type']
#         save['retrieval_list'] = retrieval_list
#         save['gold_list'] = data['evidence_list']
#         retrieval_save_list.append(save)
#
#     with open(save_file, 'w') as json_file:
#         json.dump(retrieval_save_list, json_file)

import json
from util import rm_file
from tqdm import tqdm
import argparse
from copy import deepcopy
import os
import shutil
from util import JSONReader
import openai
from typing import List, Dict
import requests
from dotenv import load_dotenv
from llama_index.embeddings.base import BaseEmbedding

from llama_index import (
    ServiceContext,
    OpenAIEmbedding,
    PromptHelper,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    set_global_service_context
)
from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding, VoyageEmbedding, InstructorEmbedding
from llama_index.postprocessor import FlagEmbeddingReranker
from llama_index.schema import QueryBundle, MetadataMode


class CustomExtractor(BaseExtractor):
    async def aextract(self, nodes) -> List[Dict]:
        metadata_list = [
            {
                "title": (
                    node.metadata["title"]
                ),
                "source": (
                    node.metadata["source"]
                ),
                "published_at": (
                    node.metadata["published_at"]
                )
            }
            for node in nodes
        ]
        return metadata_list

class DashScopeEmbedding(BaseEmbedding):
    """
    DashScope OpenAI-compatible embedding adapter for llama-index==0.9.40.

    This avoids llama-index OpenAIEmbedding's hard-coded OpenAI model enum,
    so models like text-embedding-v4 can be used.
    """

    api_key: str
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "text-embedding-v4"
    embed_batch_size: int = 10

    def _post_embeddings(self, texts: List[str]) -> List[List[float]]:
        url = self.api_base.rstrip("/") + "/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope embedding request failed: "
                f"status={response.status_code}, body={response.text}"
            )

        result = response.json()
        data = result["data"]

        data = sorted(data, key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._post_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._post_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            embeddings.extend(self._post_embeddings(batch))
        return embeddings


def get_safe_model_name(model_name: str) -> str:
    """
    Convert model name or local model path into a safe name for output/storage.
    Examples:
        sentence-transformers/all-MiniLM-L6-v2 -> all-MiniLM-L6-v2
        D:\\models\\all-MiniLM-L6-v2 -> all-MiniLM-L6-v2
    """
    normalized = model_name.replace("\\", "/")
    safe_name = normalized.split("/")[-1]
    safe_name = safe_name.replace(":", "_")
    return safe_name


def has_persisted_index(persist_dir: str) -> bool:
    """
    A simple check for whether a llama-index storage directory already exists.
    """
    return os.path.isdir(persist_dir) and len(os.listdir(persist_dir)) > 0


if __name__ == '__main__':
    load_dotenv()

    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
    dashscope_base_url = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
    openai.base_url = "your_api_base"
    voyage_api_key = os.environ.get("VOYAGE_API_KEY", "your_voyage_api_key")
    cohere_api_key = os.environ.get("COHERE_API_KEY", "your_cohere_api_key")

    parser = argparse.ArgumentParser(description="running script.")
    parser.add_argument('--retriever', type=str, required=True, help='retriever name')
    parser.add_argument('--llm', type=str, required=False, default="gpt-3.5-turbo-1106", help='LLMs')
    parser.add_argument('--rerank', action='store_true', required=False, default=False, help='if rerank')
    parser.add_argument('--topk', type=int, required=False, default=10, help='Top K')
    parser.add_argument('--chunk_size', type=int, required=False, default=256, help='chunk_size')
    parser.add_argument('--context_window', type=int, required=False, default=2048, help='context_window')
    parser.add_argument('--num_output', type=int, required=False, default=256, help='num_output')
    parser.add_argument(
        '--persist_dir',
        type=str,
        required=False,
        default=None,
        help='directory to persist/load vector index'
    )
    parser.add_argument(
        '--rebuild_index',
        action='store_true',
        required=False,
        default=False,
        help='force rebuilding index even if persisted index exists'
    )

    args = parser.parse_args()
    os.makedirs("output", exist_ok=True)
    os.makedirs("storage", exist_ok=True)

    model_name = args.retriever
    rerank = args.rerank
    top_k = args.topk
    save_model_name = get_safe_model_name(model_name)

    if args.persist_dir is None:
        persist_dir = f"storage/{save_model_name}_chunk{args.chunk_size}"
    else:
        persist_dir = args.persist_dir

    print(f"persist_dir: {persist_dir}")

    llm = OpenAI(model=args.llm, temperature=0, max_tokens=args.context_window)

    # define save file
    if rerank:
        save_file = f'output/{save_model_name}_rerank_retrieval_test.json'
    else:
        save_file = f'output/{save_model_name}_retrieval_test.json'

    rm_file(save_file)
    print(f'save_file: {save_file}')

    if model_name == "text-embedding-v4":
        if not dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is not set. Please add it to .env."
            )

        embed_model = DashScopeEmbedding(
            api_key=dashscope_api_key,
            api_base=dashscope_base_url,
            model_name=model_name,
            embed_batch_size=20,
        )
    elif 'text' in model_name:
        # OpenAI embedding models, e.g. text-embedding-ada-002
        embed_model = OpenAIEmbedding(model=model_name, embed_batch_size=10)
    elif 'Cohere' in model_name:
        embed_model = CohereEmbedding(
            cohere_api_key=cohere_api_key,
            model_name="embed-english-v3.0",
            input_type="search_query",
        )
    elif 'voyage-02' in model_name:
        embed_model = VoyageEmbedding(
            model_name='voyage-02',
            voyage_api_key=voyage_api_key
        )
    elif 'instructor' in model_name:
        embed_model = InstructorEmbedding(model_name=model_name)
    else:
        embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)

    # service context
    text_splitter = SentenceSplitter(chunk_size=args.chunk_size)
    prompt_helper = PromptHelper(
        context_window=args.context_window,
        num_output=args.num_output,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper,
    )
    set_global_service_context(service_context)

    if args.rebuild_index and os.path.isdir(persist_dir):
        print(f"rebuild_index=True, removing existing index: {persist_dir}")
        shutil.rmtree(persist_dir)

    if has_persisted_index(persist_dir):
        print(f"Loading persisted index from: {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context
        )
        print("Finish Loading Persisted Index.")
    else:
        print("No persisted index found. Building a new index...")

        reader = JSONReader()
        data = reader.load_data('dataset/corpus.json')
        # print(data[0])

        transformations = [text_splitter, CustomExtractor()]
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=data)

        print(f"Total documents: {len(data)}")
        print(f"Total nodes/chunks: {len(nodes)}")

        nodes_see = deepcopy(nodes)
        print(
            "LLM sees:\n",
            (nodes_see)[0].get_content(metadata_mode=MetadataMode.LLM),
        )
        print('Finish Loading...')

        index = VectorStoreIndex(nodes, show_progress=True)
        print('Finish Indexing...')

        print(f"Persisting index to: {persist_dir}")
        index.storage_context.persist(persist_dir=persist_dir)
        print("Finish Persisting Index.")

    with open('dataset/MultiHopRAG.json', 'r', encoding='utf-8') as file:
        query_data = json.load(file)

    if rerank:
        rerank_postprocessors = FlagEmbeddingReranker(
            model="BAAI/bge-reranker-large",
            top_n=top_k
        )

    # test retrieval quality
    retrieval_save_list = []
    print("start to retrieve...")
    for data in tqdm(query_data):
        query = data['query']

        if rerank:
            nodes_score = index.as_retriever(similarity_top_k=20).retrieve(query)
            nodes_score = rerank_postprocessors.postprocess_nodes(
                nodes_score,
                query_bundle=QueryBundle(query_str=query)
            )
        else:
            nodes_score = index.as_retriever(similarity_top_k=top_k).retrieve(query)

        retrieval_list = []
        for ns in nodes_score:
            dic = {}
            dic['text'] = ns.get_content(metadata_mode=MetadataMode.LLM)
            dic['score'] = ns.get_score()
            retrieval_list.append(dic)

        save = {}
        save['query'] = data['query']
        save['answer'] = data['answer']
        save['question_type'] = data['question_type']
        save['retrieval_list'] = retrieval_list
        save['gold_list'] = data['evidence_list']
        retrieval_save_list.append(save)

    with open(save_file, 'w', encoding='utf-8') as json_file:
        json.dump(retrieval_save_list, json_file, ensure_ascii=False)