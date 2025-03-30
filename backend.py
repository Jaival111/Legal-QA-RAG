import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import logging
import warnings
import hashlib
from pathlib import Path
import time
from requests.exceptions import ReadTimeout
import os
from huggingface_hub import login

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalRAGSystem:
    def __init__(self, cache_dir=".cache"):

        login(token=os.environ["HUGGINGFACE_TOKEN"])
        
        _ = torch.zeros(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = None
        self.llm = None
        self.tokenizer = None
        self.df = None
        self.embeddings = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_context_length = 2000
        self.full_text_cache = {}
        self.max_retries = 3
        self.timeout = 30

    def _get_cache_path(self, key):
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
    
    def _load_dataset_with_retry(self):
        """Load dataset with retry logic for timeout handling."""
        for attempt in range(self.max_retries):
            try:
                return load_dataset(
                    "opennyaiorg/InJudgements_dataset", 
                    split="train",
                    download_mode="force_redownload" if attempt > 0 else "reuse_dataset_if_exists"
                )
            except (ReadTimeout, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (attempt + 1) * 5
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the legal documents dataset."""
        logger.info("Loading Indian legal documents...")
        try:
            dataset = self._load_dataset_with_retry()
            df = pd.DataFrame(dataset)
            
            df = df[['Titles', 'Court_Name', 'Text', 'Case_Type', 'Court_Type']].rename(
                columns={'Titles': 'title', 'Court_Name': 'court', 'Text': 'text'}
            )
            
            df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.len() > 500]
            
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def initialize_models(self):
        """Initialize models with proper error handling."""
        logger.info("Initializing models...")
        
        try:
            self.embedding_model = SentenceTransformer(
                "all-mpnet-base-v2",
                device=self.device
            )
            
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

    def generate_embeddings(self, batch_size: int = 32):
        """Generate document embeddings with error checking."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized. Call initialize_models() first.")
            
        logger.info(f"Generating embeddings for {len(self.df)} documents...")
        texts = self.df['text'].tolist()
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.embedding_model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device
            )
            embeddings.append(emb.cpu().numpy())
        
        self.embeddings = np.concatenate(embeddings)

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents using semantic search."""
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call generate_embeddings() first.")
            
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        ).cpu().numpy()
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            doc = self.df.iloc[idx]
            doc_id = f"{doc['title']}_{idx}"
            self.full_text_cache[doc_id] = doc['text']
            
            results.append({
                "score": float(similarities[idx]),
                "title": doc['title'],
                "text": doc['text'][:self.max_context_length],
                "full_text_id": doc_id,
                "court": doc['court'],
                "case_type": doc['Case_Type'],
                "court_type": doc['Court_Type']
            })
        
        return results
    
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