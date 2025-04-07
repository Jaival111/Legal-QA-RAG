import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import logging
import warnings
from pathlib import Path
import time
from requests.exceptions import ReadTimeout
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalRAGSystem:
    def __init__(self, cache_dir=".cache", chroma_persist_dir=".chroma_db"):
        _ = torch.zeros(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = None
        self.llm = None
        self.tokenizer = None
        self.df = None
        self.max_context_length = 2000
        self.full_text_cache = {}
        self.max_retries = 3
        self.timeout = 60
        
        # Initialize ChromaDB
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.chroma_persist_dir.mkdir(exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_persist_dir)
        )

        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="legal_documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-mpnet-base-v2",
                device=self.device
            )
        )
        
        # Create cache directory if it doesn't exist
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chroma_persist_dir.mkdir(exist_ok=True)

    def _load_dataset_with_retry(self):
        """Load dataset with retry logic and proper authentication."""
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        for attempt in range(self.max_retries):
            try:
                return load_dataset(
                    "opennyaiorg/InJudgements_dataset",
                    split="train",
                    token=hf_token,
                    download_mode="force_redownload" if attempt > 0 else "reuse_dataset_if_exists"
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Final attempt failed: {str(e)}")
                    raise
                wait_time = (attempt + 1) * 15
                logger.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)

    def load_data(self, test_mode: bool = False, test_size: int = 1000) -> pd.DataFrame:
        """Load dataset with fresh download each time, with option to limit for testing."""
        logger.info("Downloading legal documents...")
        try:
            dataset = self._load_dataset_with_retry()
            df = pd.DataFrame(dataset)
            
            df = df[['Titles', 'Court_Name', 'Text', 'Case_Type', 'Court_Type']].rename(
                columns={'Titles': 'title', 'Court_Name': 'court', 'Text': 'text'}
            )
            df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Filter for quality documents first
            df = df.dropna(subset=['text']).loc[df['text'].str.len() > 500]
            
            # Add document IDs
            df['doc_id'] = df['title'].str[:50] + '_' + df.index.astype(str)
            
            # Limit to test_size documents if in test mode
            if test_mode:
                logger.info(f"Running in test mode - limiting to {test_size} documents")
                df = df.head(test_size)
                
            self.df = df
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def initialize_models(self):
        """Initialize models with proper error handling."""
        logger.info("Initializing models...")
        
        try:
            # We don't need to initialize the embedding model separately since ChromaDB handles it
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"
            for attempt in range(self.max_retries):
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    break
                except (ReadTimeout, ConnectionError) as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Model download attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def generate_and_store_embeddings(self, batch_size: int = 32, test_mode: bool = False, test_size: int = 1000):
        """Generate document embeddings and store in ChromaDB with error checking."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # If in test mode and we have more documents than test_size, limit them
        if test_mode and len(self.df) > test_size:
            self.df = self.df.head(test_size)
            logger.info(f"Test mode active - using first {test_size} documents for embeddings")
            
        # Check if collection already has documents
        existing_count = self.collection.count()
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing documents in ChromaDB - skipping embedding generation")
            return
            
        logger.info(f"Generating and storing embeddings for {len(self.df)} documents...")
        
        # Prepare documents for ChromaDB
        documents = self.df['text'].tolist()
        metadatas = self.df[['title', 'court', 'Case_Type', 'Court_Type']].to_dict('records')
        ids = self.df['doc_id'].tolist()
        
        # Add to ChromaDB in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents using ChromaDB's semantic search."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Process results
        retrieved_docs = []
        for i, (doc, meta, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            doc_id = f"{meta['title']}_{i}"
            self.full_text_cache[doc_id] = doc
            
            retrieved_docs.append({
                "score": 1 - distance,  # Convert distance to similarity score
                "title": meta['title'],
                "text": doc[:self.max_context_length],
                "full_text_id": doc_id,
                "court": meta['court'],
                "case_type": meta['Case_Type'],
                "court_type": meta['Court_Type']
            })
        
        return retrieved_docs
    
    def get_full_text(self, doc_id: str) -> str:
        """Retrieve full document text from cache."""
        return self.full_text_cache.get(doc_id, "Document not found")

    def generate_response(
        self,
        question: str,
        context: List[Dict],
        max_new_tokens: int = 512
    ) -> str:
        """Generate answer using Mistral with RAG context."""
        formatted_context = "\n\n".join(
            f"Document {i+1}: {doc['title']}\nCourt: {doc['court']}\n{doc['text']}"
            for i, doc in enumerate(context)
        )
        
        prompt = f"""<s>[INST] You are an Indian legal expert. Analyze the provided judgments and answer professionally.
        
        Context:
        {formatted_context}
        
        Question: {question}
        
        Provide a detailed response citing relevant legal provisions and case laws. 
        If the answer isn't clear from context, state "Not determinable from provided documents".[/INST]"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
