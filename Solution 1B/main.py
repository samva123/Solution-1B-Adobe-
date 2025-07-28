import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

import PyPDF2
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

class DocumentAnalyzer:
    def __init__(self):
        print("Loading models...")
        
        # Load lightweight sentence transformer model
        # self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)

        
        # Load lightweight text generation model for analysis
        # self.summarizer = pipeline(
        #     "summarization",
        #     model="facebook/bart-large-cnn",
        #     tokenizer="facebook/bart-large-cnn",
        #     device=-1  # CPU only
        # )
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=-1,
            local_files_only=True
        )

        print("Models loaded successfully!")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with page numbers"""
        page_texts = {}
        
        try:
            # Use PyMuPDF for better text extraction
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    page_texts[page_num + 1] = text.strip()
            doc.close()
        except Exception as e:
            print(f"Error with PyMuPDF for {pdf_path}: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            page_texts[page_num + 1] = text.strip()
            except Exception as e2:
                print(f"Error with PyPDF2 for {pdf_path}: {e2}")
        
        return page_texts
    
    def segment_text_into_sections(self, text: str, page_num: int) -> List[Dict]:
        """Segment text into logical sections"""
        sections = []
        
        # Split by common section indicators
        patterns = [
            r'\n\s*(?=[A-Z][A-Za-z\s]{10,})\n',  # Title-like patterns
            r'\n\s*\d+\.\s+[A-Z]',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]+\n',  # All caps headers
            r'\n\s*(?:Chapter|Section|Part)\s+\d+',  # Explicit chapters/sections
        ]
        
        # Try to split by patterns
        segments = [text]
        for pattern in patterns:
            new_segments = []
            for segment in segments:
                new_segments.extend(re.split(pattern, segment))
            segments = new_segments
        
        # Clean and create sections
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if len(segment) > 100:  # Minimum length for a section
                # Extract potential title (first line or first sentence)
                lines = segment.split('\n')
                title = lines[0][:100] + "..." if len(lines[0]) > 100 else lines[0]
                
                sections.append({
                    'title': title.strip(),
                    'content': segment,
                    'page_number': page_num,
                    'word_count': len(segment.split())
                })
        
        return sections
    
    def create_persona_job_embedding(self, persona: str, job: str) -> np.ndarray:
        """Create embedding for persona and job combination"""
        combined_text = f"Role: {persona}. Task: {job}"
        return self.sentence_model.encode([combined_text])[0]
    
    def rank_sections_by_relevance(self, sections: List[Dict], persona_job_embedding: np.ndarray) -> List[Dict]:
        """Rank sections by relevance to persona and job"""
        if not sections:
            return []
        
        # Create embeddings for all sections
        section_texts = [f"{section['title']} {section['content'][:500]}" for section in sections]
        section_embeddings = self.sentence_model.encode(section_texts)
        
        # Calculate similarity scores
        similarities = cosine_similarity([persona_job_embedding], section_embeddings)[0]
        
        # Add similarity scores to sections
        for i, section in enumerate(sections):
            section['relevance_score'] = float(similarities[i])
        
        # Sort by relevance score
        ranked_sections = sorted(sections, key=lambda x: x['relevance_score'], reverse=True)
        
        return ranked_sections
    
    def extract_key_subsections(self, section_content: str, persona: str, job: str) -> List[str]:
        """Extract key subsections from a section"""
        # Split into paragraphs or sentences
        sentences = re.split(r'[.!?]+', section_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        if not sentences:
            return [section_content[:500]]
        
        # Create persona-job context
        context = f"For a {persona} who needs to {job}, the most relevant information is:"
        
        # Score sentences by relevance
        persona_job_embedding = self.sentence_model.encode([context])
        sentence_embeddings = self.sentence_model.encode(sentences)
        
        similarities = cosine_similarity(persona_job_embedding, sentence_embeddings)[0]
        
        # Get top sentences
        scored_sentences = list(zip(sentences, similarities))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 most relevant subsections
        top_subsections = [sent for sent, score in scored_sentences[:3]]
        
        return top_subsections
    
    def process_collection(self, collection_path: str) -> Dict[str, Any]:
        """Process a single collection folder"""
        collection_path = Path(collection_path)
        
        # Load input.json
        input_file = collection_path / "input.json"
        if not input_file.exists():
            raise FileNotFoundError(f"input.json not found in {collection_path}")
        
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Extract persona and job
        persona = input_data['persona']['role']
        job = input_data['job_to_be_done']['task']
        
        print(f"Processing collection: {collection_path.name}")
        print(f"Persona: {persona}")
        print(f"Job: {job}")
        
        # Create persona-job embedding
        persona_job_embedding = self.create_persona_job_embedding(persona, job)
        
        # Process all PDFs
        all_sections = []
        pdfs_folder = collection_path / "pdfs"
        
        if not pdfs_folder.exists():
            raise FileNotFoundError(f"pdfs folder not found in {collection_path}")
        
        for doc_info in input_data['documents']:
            pdf_path = pdfs_folder / doc_info['filename']
            if not pdf_path.exists():
                print(f"Warning: {pdf_path} not found, skipping...")
                continue
            
            print(f"Processing: {doc_info['filename']}")
            
            # Extract text from PDF
            page_texts = self.extract_text_from_pdf(str(pdf_path))
            
            # Process each page
            for page_num, text in page_texts.items():
                sections = self.segment_text_into_sections(text, page_num)
                for section in sections:
                    section['document'] = doc_info['filename']
                    all_sections.append(section)
        
        print(f"Total sections extracted: {len(all_sections)}")
        
        # Rank sections by relevance
        ranked_sections = self.rank_sections_by_relevance(all_sections, persona_job_embedding)
        
        # Select top 5 most relevant sections
        top_sections = ranked_sections[:5]
        
        # Generate output
        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in input_data['documents']],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Process top sections
        for rank, section in enumerate(top_sections, 1):
            # Add to extracted_sections
            output["extracted_sections"].append({
                "document": section['document'],
                "section_title": section['title'],
                "importance_rank": rank,
                "page_number": section['page_number']
            })
            
            # Extract key subsections
            key_subsections = self.extract_key_subsections(
                section['content'], persona, job
            )
            
            # Add to subsection_analysis
            for subsection in key_subsections:
                output["subsection_analysis"].append({
                    "document": section['document'],
                    "refined_text": subsection.strip(),
                    "page_number": section['page_number']
                })
        
        return output
    
    def run_analysis(self, project_root: str):
        """Run analysis on all collections in the project"""
        project_root = Path(project_root)
        
        # Find all collection folders
        collection_folders = [
            folder for folder in project_root.iterdir() 
            if folder.is_dir() and folder.name.startswith('collection')
        ]
        
        if not collection_folders:
            print("No collection folders found!")
            return
        
        print(f"Found {len(collection_folders)} collections")
        
        for collection_folder in collection_folders:
            try:
                print(f"\n{'='*50}")
                print(f"Processing {collection_folder.name}")
                print(f"{'='*50}")
                
                output = self.process_collection(collection_folder)
                
                # Save output.json
                output_file = collection_folder / "output.json"
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=2)
                
                print(f"✅ Successfully processed {collection_folder.name}")
                print(f"📄 Output saved to: {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing {collection_folder.name}: {e}")
                continue

def main():
    """Main function to run the document analyzer"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <project_root_path>")
        print("Example: python main.py /path/to/project")
        return
    
    project_root = sys.argv[1]
    
    if not os.path.exists(project_root):
        print(f"Error: Project root path '{project_root}' does not exist!")
        return
    
    try:
        analyzer = DocumentAnalyzer()
        analyzer.run_analysis(project_root)
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()