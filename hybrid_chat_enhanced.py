# hybrid_chat_enhanced.py -GraphDatabase IMPROVED VERSION WITH RELEVANCE DETECTION
import json
from typing import List, Dict, Tuple
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
from functools import lru_cache
import time

# Config
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME
RELEVANCE_THRESHOLD = 0.65  # Minimum similarity score for relevant results

# Initialize clients
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region="us-east1-gcp")
    )

index = pc.Index(INDEX_NAME)
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# IMPROVEMENT 1: Caching for embeddings
@lru_cache(maxsize=100)
def embed_text(text: str) -> tuple:
    """Cached embedding generation."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return tuple(resp.data[0].embedding)

def check_query_relevance(query: str) -> Tuple[bool, str]:
    vietnam_keywords = [
        'vietnam', 'vietnamese', 'hanoi', 'ho chi minh', 'saigon',
        'hoi an', 'da nang', 'nha trang', 'hue', 'sapa', 'mekong',
        'ha long', 'dalat', 'da lat'
    ]
    
    travel_keywords = [
        'travel', 'trip', 'visit', 'tour', 'hotel', 'attraction',
        'itinerary', 'destination', 'vacation', 'holiday', 'tourist',
        'romantic', 'adventure', 'beach', 'culture', 'food', 'activity',
        'restaurant', 'stay', 'accommodation', 'sightseeing'
    ]
    
    query_lower = query.lower()
    
    # Check if query mentions Vietnam
    has_vietnam = any(keyword in query_lower for keyword in vietnam_keywords)
    
    # Check if query is travel-related
    has_travel = any(keyword in query_lower for keyword in travel_keywords)
    
    # Strong relevance: mentions Vietnam
    if has_vietnam:
        return True, "vietnam_specific"
    
    # Moderate relevance: travel-related without Vietnam mention
    # Will check vector similarity scores
    if has_travel:
        return True, "travel_general"
    
    # Low relevance: neither Vietnam nor travel
    return False, "not_relevant"

def pinecone_query(query_text: str, top_k=TOP_K):
    try:
        vec = list(embed_text(query_text))
        res = index.query(vector=vec, top_k=top_k, include_metadata=True)
        matches = res["matches"]
        
        # Check if results are actually relevant
        if matches:
            max_score = matches[0].get("score", 0)
            print(f"Retrieved {len(matches)} results (top score: {max_score:.3f})")
            return matches, max_score
        return [], 0.0
    except Exception as e:
        print(f"Pinecone error: {e}")
        return [], 0.0

def fetch_graph_context_enhanced(node_ids: List[str]):
    if not node_ids:
        return {"nodes": [], "relationships": [], "cities": set(), "tags": set()}
    
    context = {"nodes": [], "relationships": [], "cities": set(), "tags": set()}
    
    with driver.session() as session:
        for nid in node_ids:
            # Get full node details
            node_q = """
            MATCH (n:Entity {id: $nid})
            RETURN n.id AS id, n.name AS name, n.type AS type,
                   n.description AS description, n.city AS city,
                   n.tags AS tags, n.best_time_to_visit AS best_time
            """
            for rec in session.run(node_q, nid=nid):
                context["nodes"].append(dict(rec))
                if rec["city"]: context["cities"].add(rec["city"])
                if rec["tags"]: context["tags"].update(rec["tags"])
            
            # Multi-hop relationships
            rel_q = """
            MATCH (n:Entity {id: $nid})-[r*1..2]-(m:Entity)
            WITH n, m LIMIT 15
            RETURN m.id AS id, m.name AS name, m.type AS type,
                   m.description AS desc, m.city AS city
            """
            for rec in session.run(rel_q, nid=nid):
                context["relationships"].append(dict(rec))
    
    print(f"Retrieved {len(context['nodes'])} nodes, {len(context['relationships'])} relationships")
    return context

def expand_query(query: str) -> List[str]:
    expansions = [query]
    keywords = {
        "romantic": ["couples", "honeymoon"],
        "adventure": ["trekking", "outdoor"],
        "beach": ["coastal", "resort"],
        "culture": ["heritage", "history"]
    }
    for key, exps in keywords.items():
        if key in query.lower():
            expansions.append(f"{query} {exps[0]}")
            break
    return expansions[:2]

def build_enhanced_prompt(query: str, matches: List, context: Dict):
    """Better structured prompts with examples."""
    system = """You are an expert Vietnam travel advisor.

Provide:
- Detailed recommendations with specific locations (include IDs)
- Practical travel tips
- Structured itineraries if requested
- Cultural insights

Format responses clearly with headers and bullet points."""

    vec_ctx = "**Top Matches:**\n"
    for i, m in enumerate(matches[:5], 1):
        meta = m["metadata"]
        vec_ctx += f"{i}. {meta.get('name')} (ID: `{m['id']}`)\n"
        vec_ctx += f"   Type: {meta.get('type')} | City: {meta.get('city')}\n"
        vec_ctx += f"   Tags: {', '.join(meta.get('tags', []))}\n\n"
    
    graph_ctx = "**Related Places:**\n"
    for node in context["nodes"][:5]:
        graph_ctx += f"- **{node['name']}** ({node['type']})\n"
        graph_ctx += f"  {node['description'][:150]}...\n\n"
    
    insights = f"**Coverage:** {len(context['cities'])} cities, themes: {', '.join(list(context['tags'])[:5])}"
    
    user_msg = f"""Question: "{query}"

{vec_ctx}

{graph_ctx}

{insights}

Provide a comprehensive answer with:
1. Overview of recommendations
2. Specific places with IDs
3. Day-by-day itinerary if applicable
4. Practical tips"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ]

def build_general_prompt(query: str):
    system = """You are a helpful AI assistant. When asked questions outside of Vietnam travel, 
provide accurate, concise answers using your general knowledge. 

If the question is completely unrelated to travel, briefly answer it and then 
suggest how you can help with Vietnam travel planning."""

    user_msg = f"""Question: "{query}"

Please provide a brief, accurate answer to this question. 

After answering, mention: "I'm primarily designed to help with Vietnam travel planning. 
Feel free to ask me about destinations, itineraries, hotels, or activities in Vietnam!"
"""
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ]

def call_chat(messages):
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def generate_response(query: str):
    # Check query relevance
    is_relevant, relevance_type = check_query_relevance(query)
    
    if not is_relevant:
        print("Query not related to Vietnam travel - using general knowledge")
        prompt = build_general_prompt(query)
        return call_chat(prompt), "general"
    
    # For travel queries, proceed with RAG system
    print(f"Detected {relevance_type} query - searching knowledge base")
    
    # Query expansion
    expanded = expand_query(query)
    all_matches = []
    max_scores = []
    
    for exp_q in expanded:
        matches, score = pinecone_query(exp_q, TOP_K)
        all_matches.extend(matches)
        max_scores.append(score)
    
    # Check if vector results are actually relevant
    best_score = max(max_scores) if max_scores else 0
    
    if best_score < RELEVANCE_THRESHOLD and relevance_type != "vietnam_specific":
        print(f"Low relevance score ({best_score:.3f}) - using general knowledge")
        prompt = build_general_prompt(query)
        return call_chat(prompt), "general"
    
    # Deduplicate
    seen = set()
    matches = []
    for m in all_matches:
        if m["id"] not in seen:
            seen.add(m["id"])
            matches.append(m)
    matches = matches[:TOP_K]
    
    if not matches:
        # No matches found - use general knowledge
        prompt = build_general_prompt(query)
        return call_chat(prompt), "general"
    
    # Get graph context
    ids = [m["id"] for m in matches]
    context = fetch_graph_context_enhanced(ids)
    
    # Generate response using RAG
    prompt = build_enhanced_prompt(query, matches, context)
    answer = call_chat(prompt)
    
    return answer, "rag"

# Interactive loop
def interactive_chat():
    print("Voyage AT")
    print("\nAsk about Vietnam destinations, itineraries, and activities")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\nYour question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            print("\nThank you for using Voyage AI!")
            break

        print("\nProcessing query...")
        start = time.time()
        
        # Generate response with relevance checking
        answer, response_type = generate_response(query)
        
        print("\n" + "=" * 60)
        if response_type == "general":
            print("[General Knowledge Response]")
        print(answer)
        print("=" * 60)
        print(f"Response time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    try:
        interactive_chat()
    finally:
        driver.close()
