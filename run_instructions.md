# Complete Project Setup and Execution Guide - OFFLINE VERSION

## Project Structure
```
document-intelligence-system/
├── offline_main.py           # Main processing script (offline)
├── requirements.txt          # Minimal Python dependencies  
├── Dockerfile               # Container configuration
├── approach_explanation.md  # Methodology documentation
├── README.md               # This file
└── data/                   # Data directory (mounted as volume)
    ├── collection1/
    │   ├── input.json      # Collection configuration
    │   ├── pdfs/           # PDF documents folder
    │   │   ├── doc1.pdf
    │   │   └── doc2.pdf
    │   └── output.json     # Generated output (created by system)
    ├── collection2/
    │   ├── input.json
    │   ├── pdfs/
    │   └── output.json
    └── ...
```

## Setup Instructions

### Option 1: Docker Setup (Recommended)

1. **Build the Docker image:**
```bash
docker build -t document-intelligence .
```

2. **Prepare your data structure:**
```bash
mkdir -p data/collection1/pdfs
mkdir -p data/collection2/pdfs
# Add more collections as needed
```

3. **Create input.json files:**
For each collection, create an `input.json` file with this format:
```json
{
    "challenge_info": {
        "challenge_id": "round_1b_003",
        "test_case_name": "create_manageable_forms",
        "description": "Creating manageable forms"
    },
    "documents": [
        {
            "filename": "Learn Acrobat - Create and Convert_1.pdf",
            "title": "Learn Acrobat - Create and Convert_1"
        },
        {
            "filename": "Learn Acrobat - Create and Convert_2.pdf",
            "title": "Learn Acrobat - Create and Convert_2"
        }
    ],
    "persona": {
        "role": "HR professional"
    },
    "job_to_be_done": {
        "task": "Create and manage fillable forms for onboarding and compliance."
    }
}
```

4. **Place PDF files:**
Copy your PDF documents to the respective `pdfs/` folders in each collection.

5. **Run the system (Windows PowerShell):**
```powershell
docker run -v ${pwd}/data:/app/data document-intelligence
```

**Run the system (Linux/Mac):**
```bash
docker run -v $(pwd)/data:/app/data document-intelligence
```

### Option 2: Local Python Setup

1. **Install Python 3.8+ and pip**

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the system:**
```bash
python offline_main.py data/
```

## Key Features of Offline Version

### 🚀 **No Internet Required**
- **Pure Offline Processing**: No model downloads during execution
- **Minimal Dependencies**: Only PyPDF2, NumPy, and scikit-learn
- **Fast Startup**: No model loading delays
- **Reliable**: Works in completely isolated environments

### 🧠 **Enhanced Intelligence Without LLMs**
- **Advanced TF-IDF**: Enhanced with persona-specific keyword weighting
- **Pattern Recognition**: Sophisticated section detection using multiple regex patterns
- **Keyword Matching**: Persona-aware keyword dictionaries for different roles
- **Action Recognition**: Identifies procedural content with action words

### 📊 **Smart Scoring System**
- **Multi-factor Relevance**: Combines TF-IDF, keyword matching, and content analysis
- **Length Normalization**: Prefers substantial, informative content
- **Position Weighting**: Values first and last sentences appropriately
- **Pattern Bonuses**: Rewards instructional and procedural content

## Expected Output Format

Same as before - the system generates an `output.json` file in each collection folder with the exact format specified.

## System Features

### 🚀 **Performance**
- **CPU-only processing** (no GPU required)
- **Under 1GB total** (minimal dependencies)
- **Processes 3-5 documents in under 30 seconds**
- **Cross-platform compatibility**
- **No internet access required**

### 🧠 **Intelligence Features**
- **Persona-Specific Keywords**: Built-in dictionaries for HR, researchers, students, analysts
- **Enhanced Section Detection**: Multiple pattern recognition techniques
- **Content Quality Assessment**: Prioritizes actionable, procedural content
- **Smart Text Extraction**: Rule-based summarization with priority patterns

### 🔧 **Flexibility**
- **Works Completely Offline**: No external dependencies
- **Lightweight**: Minimal resource requirements
- **Robust**: Handles various PDF formats and content types
- **Fast Processing**: Quick startup and execution

## Troubleshooting

### Common Issues:

1. **"input.json not found"**
   - Ensure each collection folder has an `input.json` file
   - Check the JSON format is valid

2. **"PDF not found"**
   - Verify PDF filenames in `input.json` match actual files in `pdfs/` folder
   - Check file permissions

3. **Docker permission issues (Windows):**
   ```powershell
   # Use full path instead of ${pwd} if having issues
   docker run -v C:\full\path\to\your\data:/app/data document-intelligence
   ```

4. **Performance issues:**
   - The offline version should be much faster since no model loading
   - Reduce number of documents per collection if needed
   - Ensure sufficient RAM (recommended: 2GB+)

## Why This Offline Approach Works

### 🎯 **Smart Text Analysis Without LLMs**
- **Enhanced TF-IDF**: Goes beyond basic term frequency with domain knowledge
- **Persona Intelligence**: Uses curated keyword dictionaries for different roles
- **Pattern Recognition**: Identifies important content types (instructions, procedures, definitions)
- **Context Awareness**: Understands document structure and content hierarchy

### 📈 **Performance Benefits**
- **Instant Startup**: No model downloading or loading delays
- **Consistent Performance**: Same speed regardless of internet connectivity  
- **Lower Resource Usage**: Uses much less memory and CPU
- **Reliable**: Works in any environment, including air-gapped systems

The offline version maintains high accuracy through intelligent text analysis techniques while being completely self-contained and fast. system generates an `output.json` file in each collection folder:

```json
{
    "metadata": {
        "input_documents": [
            "Learn Acrobat - Create and Convert_1.pdf",
            "Learn Acrobat - Create and Convert_2.pdf"
        ],
        "persona": "HR professional",
        "job_to_be_done": "Create and manage fillable forms for onboarding and compliance.",
        "processing_timestamp": "2025-07-10T15:34:33.350102"
    },
    "extracted_sections": [
        {
            "document": "Learn Acrobat - Fill and Sign.pdf",
            "section_title": "Change flat forms to fillable (Acrobat Pro)",
            "importance_rank": 1,
            "page_number": 12
        }
    ],
    "subsection_analysis": [
        {
            "document": "Learn Acrobat - Fill and Sign.pdf",
            "refined_text": "To create an interactive form, use the Prepare Forms tool...",
            "page_number": 12
        }
    ]
}
```

## System Features

### 🚀 **Performance**
- CPU-only processing (no GPU required)
- Processes 3-5 documents in under 60 seconds
- Model size under 1GB total
- Cross-platform compatibility

### 🧠 **Intelligence**
- Uses sentence transformers for semantic understanding
- Persona-aware content ranking
- Context-sensitive section extraction
- No internet access required during execution

### 🔧 **Flexibility**
- Supports any document domain
- Works with diverse persona types
- Handles various job-to-be-done scenarios
- Batch processes multiple collections

## Troubleshooting

### Common Issues:

1. **"input.json not found"**
   - Ensure each collection folder has an `input.json` file
   - Check the JSON format is valid

2. **"PDF not found"**
   - Verify PDF filenames in `input.json` match actual files in `pdfs/` folder
   - Check file permissions

3. **Memory issues**
   - Reduce number of documents per collection
   - Ensure sufficient RAM (recommended: 4GB+)

4. **Docker permission issues**
   - On Linux/Mac: `sudo docker run -v $(pwd)/data:/app/data document-intelligence`
   - On Windows: Use full path instead of `$(pwd)`

### Performance Optimization:

- For faster processing, use smaller PDF files (< 50 pages each)
- Process collections sequentially rather than in parallel
- Close other memory-intensive applications

## Testing the System

Use the provided sample data structure to test:

1. Create a test collection with the HR professional example
2. Add sample Acrobat PDF files
3. Run the system and verify output format
4. Check that extracted sections are relevant to HR form creation tasks

The system will automatically rank sections based on semantic relevance to the HR professional's need to create fillable forms, providing the most useful content for onboarding and compliance scenarios.