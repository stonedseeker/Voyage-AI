# Voyage AI 🌏

> Intelligent Vietnam travel assistant powered by hybrid RAG architecture

An AI-powered travel planning system that combines semantic search with graph relationships to generate personalized Vietnam itineraries.

## Quick Start
```bash
# Clone and setup
git clone https://github.com/stonedseeker/Voyage-AI.git
cd Voyage-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure (add your API keys)
cp config_example.py config.py # Edit config.py with your credentials

# Load data
python load_to_neo4j.py
python pinecone_upload.py

# Run
python hybrid_chat_enhanced.py
```

## What It Does

Ask natural questions, get intelligent answers:
```
You: "Romantic 4-day Vietnam itinerary"
Voyage AI: [Generates detailed day-by-day plan with hotels, 
            attractions, and travel routes]
```

## Architecture
```
Query → Vector Search (Pinecone) → Graph Relationships (Neo4j) → GPT-4 → Itinerary
```

## Requirements

- Python 3.10+
- [Neo4j Aura](https://neo4j.com/cloud/aura/) (free tier)
- [OpenAI API](https://platform.openai.com/) key
- [Pinecone](https://www.pinecone.io/) (free tier)

## Dataset

- 360 locations across Vietnam
- 10 major cities
- 700+ relationships (hotels, attractions, activities)
- 1536-dim vector embeddings

## Tech Stack

![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-purple)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)

## Example Queries

- "Romantic places in Vietnam"
- "Adventure activities in Sapa"
- "Best beaches for families"
- "4-day cultural tour starting from Hanoi"

## Contributing

Contributions welcome! Open an issue or submit a PR.

---
