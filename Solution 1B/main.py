#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence System
Main processing script for extracting and ranking document sections
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import re
from collections import defaultdict

class DocumentIntelligenceSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the system with a lightweight sentence transformer"""
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer(model_name)
        
        print("Loading summarization pipeline...")
        # Use a lightweight summarization model
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=-1  # CPU only
        )
        
        print("Models loaded successfully!")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF by page"""
        page_texts = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        page_texts[page_num] = text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return page_texts
    
    def identify_sections(self, text: str, page_num: int) -> List[Dict]:
        """Identify sections within a page using pattern matching"""
        sections = []
        
        # Common section patterns
        patterns = [
            r'^([A-Z][A-Za-z\s&-]{3,50})\n',  # Title case headings
            r'^\d+\.\s*([A-Za-z][A-Za-z\s&-]{3,50})\n',  # Numbered sections
            r'^([A-Z\s]{3,30})\n',  # ALL CAPS headings
            r'^\*\*([A-Za-z][A-Za-z\s&-]{3,50})\*\*',  # Bold headings
            r'^#\s*([A-Za-z][A-Za-z\s&-]{3,50})\n',  # Markdown-style headings
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches section pattern
            is_section_header = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_section and current_content:
                        sections.append({
                            'title': current_section,
                            'content': '\n'.join(current_content),
                            'page_number': page_num,
                            'start_line': i - len(current_content),
                            'end_line': i
                        })
                    
                    current_section = match.group(1).strip()
                    current_content = []
                    is_section_header = True
                    break
            
            if not is_section_header and current_section:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content),
                'page_number': page_num,
                'start_line': len(lines) - len(current_content),
                'end_line': len(lines)
            })
        
        # If no sections found, treat entire page as one section
        if not sections and text.strip():
            # Extract a title from first meaningful line
            first_lines = [line.strip() for line in lines[:5] if line.strip()]
            title = first_lines[0] if first_lines else f"Page {page_num} Content"
            if len(title) > 100:
                title = title[:97] + "..."
            
            sections.append({
                'title': title,
                'content': text,
                'page_number': page_num,
                'start_line': 0,
                'end_line': len(lines)
            })
        
        return sections
    
    def create_persona_query(self, persona: str, job_to_be_done: str) -> str:
        """Create a query representing the persona's needs"""
        return f"{persona} needs to {job_to_be_done}. Focus on relevant tools, procedures, and best practices."
    
    def calculate_relevance_score(self, section_content: str, persona_query: str) -> float:
        """Calculate relevance score using sentence embeddings"""
        try:
            # Create embeddings
            section_embedding = self.sentence_model.encode([section_content])
            query_embedding = self.sentence_model.encode([persona_query])
            
            # Calculate cosine similarity
            similarity = np.dot(section_embedding[0], query_embedding[0]) / (
                np.linalg.norm(section_embedding[0]) * np.linalg.norm(query_embedding[0])
            )
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0
    
    def extract_key_subsections(self, content: str, max_length: int = 200) -> str:
        """Extract key information from content using summarization"""
        try:
            if len(content) < 50:
                return content
            
            # Split into sentences and select most relevant ones
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            if not sentences:
                return content[:max_length]
            
            # If content is short enough, return as is
            if len(content) <= max_length:
                return content
            
            # Use extractive approach for longer content
            if len(sentences) <= 3:
                return '. '.join(sentences[:3])
            
            # Select key sentences based on position and length
            key_sentences = []
            
            # Always include first sentence if it's informative
            if sentences and len(sentences[0]) > 20:
                key_sentences.append(sentences[0])
            
            # Add middle sentences that contain important keywords
            important_keywords = ['create', 'form', 'tool', 'enable', 'fill', 'sign', 'field', 'select', 'click', 'choose']
            for sentence in sentences[1:-1]:
                if any(keyword in sentence.lower() for keyword in important_keywords):
                    key_sentences.append(sentence)
                    if len('. '.join(key_sentences)) > max_length:
                        break
            
            # If we have room, add the last sentence
            if len(key_sentences) < 3 and sentences:
                key_sentences.append(sentences[-1])
            
            result = '. '.join(key_sentences)
            if len(result) > max_length:
                result = result[:max_length-3] + "..."
            
            return result
            
        except Exception as e:
            print(f"Error extracting subsections: {e}")
            return content[:max_length]
    
    def process_documents(self, collection_path: str) -> Dict[str, Any]:
        """Process all documents in a collection"""
        collection_path = Path(collection_path)
        
        # Load input configuration
        input_file = collection_path / "input.json"
        if not input_file.exists():
            raise FileNotFoundError(f"input.json not found in {collection_path}")
        
        with open(input_file, 'r') as f:
            config = json.load(f)
        
        # Extract configuration
        persona = config["persona"]["role"]
        job_to_be_done = config["job_to_be_done"]["task"]
        documents = config["documents"]
        
        print(f"Processing {len(documents)} documents for {persona}")
        print(f"Job to be done: {job_to_be_done}")
        
        # Create persona query
        persona_query = self.create_persona_query(persona, job_to_be_done)
        print(f"Persona query: {persona_query}")
        
        # Process each document
        all_sections = []
        pdfs_path = collection_path / "pdfs"
        
        for doc_info in documents:
            filename = doc_info["filename"]
            pdf_path = pdfs_path / filename
            
            if not pdf_path.exists():
                print(f"Warning: {pdf_path} not found")
                continue
            
            print(f"Processing {filename}...")
            
            # Extract text by page
            page_texts = self.extract_text_from_pdf(str(pdf_path))
            
            # Process each page
            for page_num, text in page_texts.items():
                sections = self.identify_sections(text, page_num)
                
                for section in sections:
                    # Calculate relevance score
                    relevance_score = self.calculate_relevance_score(
                        section['content'], persona_query
                    )
                    
                    section_data = {
                        'document': filename,
                        'section_title': section['title'],
                        'page_number': page_num,
                        'content': section['content'],
                        'relevance_score': relevance_score
                    }
                    
                    all_sections.append(section_data)
        
        # Sort by relevance score
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Select top 5 sections
        top_sections = all_sections[:5]
        
        # Create extracted sections output
        extracted_sections = []
        for i, section in enumerate(top_sections, 1):
            extracted_sections.append({
                "document": section['document'],
                "section_title": section['section_title'],
                "importance_rank": i,
                "page_number": section['page_number']
            })
        
        # Create subsection analysis
        subsection_analysis = []
        for section in top_sections:
            refined_text = self.extract_key_subsections(section['content'])
            subsection_analysis.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
        
        # Create output structure
        output = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in documents],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output

def main():
    """Main function to process collections"""
    if len(sys.argv) != 2:
        print("Usage: python main.py <collections_root_path>")
        sys.exit(1)
    
    collections_root = Path(sys.argv[1])
    
    if not collections_root.exists():
        print(f"Collections root path {collections_root} does not exist")
        sys.exit(1)
    
    # Initialize the system
    system = DocumentIntelligenceSystem()
    
    # Process each collection
    for collection_dir in collections_root.iterdir():
        if collection_dir.is_dir() and collection_dir.name.startswith('collection'):
            print(f"\n{'='*50}")
            print(f"Processing {collection_dir.name}")
            print(f"{'='*50}")
            
            try:
                # Process the collection
                output = system.process_documents(collection_dir)
                
                # Save output
                output_file = collection_dir / "output.json"
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=4)
                
                print(f"✅ Successfully processed {collection_dir.name}")
                print(f"📄 Output saved to {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing {collection_dir.name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()