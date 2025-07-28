#!/usr/bin/env python3
"""
Script to pre-download all required models for offline execution
Run this script once with internet connection before running the main analysis
"""

import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

def download_models():
    """Download all required models for offline execution"""
    
    print("🔄 Downloading models for offline execution...")
    print("This may take a few minutes on first run...\n")
    
    try:
        # Download sentence transformer model
        print("📥 Downloading sentence transformer model (all-MiniLM-L6-v2)...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model downloaded successfully")
        
        # Download summarization model
        print("📥 Downloading summarization model (facebook/bart-large-cnn)...")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=-1  # CPU only
        )
        print("✅ Summarization model downloaded successfully")
        
        print("\n🎉 All models downloaded successfully!")
        print("You can now run the analysis offline using: python main.py <project_path>")
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print("Please check your internet connection and try again.")
        return False
    
    return True

if __name__ == "__main__":
    download_models()