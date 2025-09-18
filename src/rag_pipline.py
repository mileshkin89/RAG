# query.py

import json
import time
from chromadb.utils import embedding_functions

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

from db.client import chromadb_client
from config import config

review_template_str = """
    Your task is to answer user questions about dishes from different cuisines. 
    Don't answer questions on other topics. 
    Use only contextual information when answering. 
    If you don't know the answer, say you don't know. 
    Answer in a friendly manner, like a chef.


    CONTEXT:
    {context}
"""


def get_context(query: str):
    start = time.perf_counter()
    query_embeddings = embedding_func([query])

    context = collection.query(
        query_embeddings=query_embeddings,
        n_results=5,
        include=["documents", "distances", "metadatas"]
    )
    context_time = time.perf_counter() - start
    print(f"'get_context'-func completed in {context_time:.3f}s")
    return context


start = time.perf_counter()
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.EMBEDDER_NAME
)
collection = chromadb_client.get_collection(name=config.COLLECTION_NAME)

query = "Tell me how to change tires in a car."

context = get_context(query)

print(json.dumps(context, indent=4, ensure_ascii=False))
#print(context)

context_time_total = time.perf_counter() - start
print(f"Get context in {context_time_total:.3f}s")


output_parser = StrOutputParser()
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["query"],
        template="{query}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "query"],
    messages=messages,
)

review_chain = review_prompt_template | chat_model | output_parser

result = review_chain.invoke({"context": context, "query": query})

print(result)

time_total = time.perf_counter() - start
print(f"Total time query completed in {time_total:.3f}s")