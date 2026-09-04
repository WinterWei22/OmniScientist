import requests
import json
from typing import List
import os
import pickle as pkl
import asyncio
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .utils import extract_and_convert_dict, exact_match_entity_type


class BGEM3Embedding:
    URL = "https://api.siliconflow.cn/v1/embeddings"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def request_embedding(self, input_: str):
        payload = {
            "model": "BAAI/bge-large-zh-v1.5",
            "input": input_,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.request("POST", self.URL, json=payload, headers=headers)
            result = json.loads(response.text)
        except Exception as e:
            print(f"Error requesting embedding: {e}")
            result = {}
        return result
    
    def embed_query(self, query: str):
        response = self.request_embedding(query)
        try:
            embedding = response['data'][0]['embedding']
        except Exception as e:
            print(f"Error embedding query: {e}")
            embedding = None
        return embedding
    
    def embed_documents(self, query_list: List[str]):
        response = self.request_embedding(query_list)
        try:
            embedding_list = [each['embedding'] for each in response['data']]
        except Exception as e:
            print(f"Error embedding documents: {e}")
            embedding_list = None
        return embedding_list


class ToolEmbeddingRetriever:
    
    def __init__(self, llm, mcp_tool_client, embedding_api_key: str, embedding_cache: str, available_tools: list = None):
        self.llm = llm
        self.mcp_tool_client = mcp_tool_client
        self.mcp_tool_map = mcp_tool_client.mcp_tool_map
        self.embedding_model = BGEM3Embedding(embedding_api_key)
        self._load_tool_embedding_cache(embedding_cache)
        self.tool_name_list = list(mcp_tool_client.mcp_tool_map.keys())
        if available_tools:
            self.tool_name_list = [tool_name for tool_name in self.tool_name_list if tool_name in available_tools]
        # Keep embeddings aligned with tool_name_list. The cache can contain stale
        # tools from previous runs, and dict insertion order is not a reliable
        # index for the current MCP tool list.
        self.tool_name_list = [
            tool_name
            for tool_name in self.tool_name_list
            if tool_name in self.tool_embedding_cache
        ]
        self.all_tool_embedding = np.array(
            [self.tool_embedding_cache[tool_name] for tool_name in self.tool_name_list]
        )
        
    def _load_tool_embedding_cache(self, embedding_cache_path: str):
        # Create directory first if it doesn't exist
        if not os.path.exists(os.path.dirname(embedding_cache_path)):
            os.makedirs(os.path.dirname(embedding_cache_path))
            
        # Load existing cache if available
        if os.path.exists(embedding_cache_path):
            with open(embedding_cache_path, 'rb') as f:
                self.tool_embedding_cache = pkl.load(f)
        else:
            self.tool_embedding_cache = {}
            
        cached_embedding_num = len(self.tool_embedding_cache)
        
        # Check if mcp_tool_map is initialized
        if not hasattr(self, 'mcp_tool_map'):
            print("Warning: mcp_tool_map not initialized yet")
            return
            
        for tool_name, tool in tqdm(self.mcp_tool_map.items()):
            if tool_name in self.tool_embedding_cache:
                continue
                
            # Check if tool has description
            if not hasattr(tool, 'description'):
                print(f"Warning: tool {tool_name} has no description")
                continue
                
            tool_func_desc = tool.description.split('Args')[0]
            embedding = self.embedding_model.embed_query(tool_func_desc)
            
            # Only cache if embedding was successful
            if embedding is not None:
                self.tool_embedding_cache[tool_name] = embedding
            else:
                print(f"Warning: Failed to generate embedding for {tool_name}")
        
        # Save cache if new embeddings were added
        if len(self.tool_embedding_cache) > cached_embedding_num:
            try:
                if not os.path.exists(os.path.dirname(embedding_cache_path)):
                    os.makedirs(os.path.dirname(embedding_cache_path))
                with open(embedding_cache_path, 'wb') as f:
                    pkl.dump(self.tool_embedding_cache, f)
                print(f"Tool embedding cache saved to {embedding_cache_path}")
            except Exception as e:
                print(f"Error saving embedding cache: {e}")
    
    def retrieve_tools(self, query: str, top_k: int = 5, explain_item: bool = False) -> List[str]:
        """
        Simple RAG. Retrieve the most relevant tools for a given query using embedding similarity.
        
        Args:
            query: The input query string
            top_k: Number of most relevant tools to return
            explain_item: Whether to explain the item
            
        Returns:
            List of tool names sorted by relevance
        """
        # Get query embedding
        if explain_item:
            query = self.explain_item(query)

        if len(self.tool_name_list) == 0 or len(self.all_tool_embedding) == 0:
            return [], []
            
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Check for NaN values in embeddings
        if np.isnan(query_embedding).any() or np.isnan(self.all_tool_embedding).any():
            raise ValueError("Embedding contains NaN, please check API response.")
            
        # Calculate cosine similarity scores
        similarities = cosine_similarity(query_embedding, self.all_tool_embedding)[0]
        _tools = list(zip(self.tool_name_list, similarities))
            
        # Sort tools by similarity score and get top k
        sorted_tools = sorted(_tools, key=lambda x: x[1], reverse=True)
        top_k_tools, top_k_scores = zip(*sorted_tools[:top_k])
        
        return list(top_k_tools), list(top_k_scores)
    
    def explain_item(self, item: str):
        """
        Explain the item that is going to be searched. Make it better to do tool retrieval.
        
        Args:
            item: The item to be explained
            
        Returns:
            str: The explanation of the item
        """
        
        prompt = f"""
        You are an experienced disease biologist, please briefly explain what this is: {item}

        Note: 1. Please summarize in one paragraph; 2. Do not use bullet points; 3. Keep the answer under 100 words.
        """
        result = self.llm.invoke(prompt)
        return result.content
    
    def batch_explain_item(self, item_list: list):
        """
        Batch explain the items that are going to be searched. Make it better to do tool retrieval.
        """
        prompt = f"""
        ## Task Description
        You are an experienced disease biologist, please briefly explain what these are: {item_list}

        ## Notes
        1. Please summarize in one paragraph; 
        2. Do not use bullet points; 
        3. Keep the answer under 100 words.

        ## Output Format
        Please provide your analysis in the following JSON format:
        ```json
        {{
            "item1": "[explanation1]",
            "item2": "[explanation2]",
            ...
        }}
        ```
        """
        result = self.llm.invoke(prompt)
        item_explanation_dict = extract_and_convert_dict(result.content)
        item_explanation_dict = {
            exact_match_entity_type(item_in_result, item_list): explanation for item_in_result, explanation in item_explanation_dict.items()
        }
        
        return item_explanation_dict

    def retrieve_tools_from_candidates(self, query: str, candidate_tools: List[str], top_k: int = 5, explain_item: bool = False) -> List[str]:
        """
        Retrieve the most relevant tools from a list of candidate tools for a given query using embedding similarity.
        
        Args:
            query: The input query string
            candidate_tools: List of candidate tool names
            top_k: Number of most relevant tools to return
            explain_item: Whether to explain the item
            
        Returns:
            List of tool names sorted by relevance
        """
        if explain_item:
            query = self.explain_item(query)
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarity scores
        tool_embeddings = []
        available_tools = []
        for tool_name in candidate_tools:
            if tool_name in self.tool_embedding_cache.keys():
                tool_embeddings.append(self.tool_embedding_cache[tool_name])
                available_tools.append(tool_name)
        
        if len(tool_embeddings) == 0:
            return [], []
        
        tool_embeddings = np.array(tool_embeddings)
        
        # Check for NaN values in embeddings
        if np.isnan(query_embedding).any() or np.isnan(tool_embeddings).any():
            raise ValueError("Embedding contains NaN, please check API response.")
            
        similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
        _tools = list(zip(available_tools, similarities))
        
        # Sort tools by similarity score and get top k
        sorted_tools = sorted(_tools, key=lambda x: x[1], reverse=True)
        top_k_tools, top_k_scores = zip(*sorted_tools[:top_k])
        
        return list(top_k_tools), list(top_k_scores)
