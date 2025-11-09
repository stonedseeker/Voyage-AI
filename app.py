# app.py
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from neo4j import GraphDatabase
import config
from functools import lru_cache
import time
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Vietnam Travel Assistant",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Vietnam-inspired aquatic theme
st.markdown("""
<style>
    /* Main theme colors - Vietnamese ocean blues */
    :root {
        --primary-blue: #0077BE;
        --ocean-blue: #006494;
        --aqua-light: #13A5B1;
        --seafoam: #4FC3C3;
        --sand: #F5E6D3;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0077BE 0%, #13A5B1 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 119, 190, 0.2);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .subtitle {
        color: #E8F4F8;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        border: 2px solid #13A5B1;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #E8F4F8 0%, #D4EBF2 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 1rem 0;
        border-left: 4px solid #0077BE;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 1rem 0;
        border-left: 4px solid #13A5B1;
    }
    
    .message-label {
        color: #006494;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0077BE 0%, #006494 100%);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        color: white;
    }
    
    /* Stats boxes */
    .stat-box {
        background: linear-gradient(135deg, #13A5B1 0%, #4FC3C3 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Search status */
    .search-status {
        background: #E8F4F8;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #13A5B1;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #006494;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #0077BE 0%, #13A5B1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 119, 190, 0.3);
    }
    
    /* Input field */
    .stTextInput input {
        border: 2px solid #13A5B1;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput input:focus {
        border-color: #0077BE;
        box-shadow: 0 0 0 3px rgba(0, 119, 190, 0.1);
    }
    
    /* Example queries */
    .example-query {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid #13A5B1;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #006494;
    }
    
    .example-query:hover {
        background: #E8F4F8;
        border-color: #0077BE;
        transform: translateX(5px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #13A5B1;
        border-top: 2px solid #E8F4F8;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #E8F4F8;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #13A5B1;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #0077BE;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'search_stats' not in st.session_state:
    st.session_state.search_stats = {}

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize API clients - cached for performance."""
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    neo4j_driver = GraphDatabase.driver(
        config.NEO4J_URI, 
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    return openai_client, pinecone_index, neo4j_driver

client, index, driver = init_clients()

# Core functions from enhanced version
@lru_cache(maxsize=100)
def embed_text(text: str) -> tuple:
    """Generate cached embeddings."""
    resp = client.embeddings.create(
        model="text-embedding-3-small", 
        input=[text]
    )
    return tuple(resp.data[0].embedding)

def pinecone_query(query_text: str, top_k=5):
    """Query Pinecone with error handling."""
    try:
        vec = list(embed_text(query_text))
        res = index.query(vector=vec, top_k=top_k, include_metadata=True)
        return res["matches"]
    except Exception as e:
        st.error(f"Pinecone error: {e}")
        return []

def fetch_graph_context(node_ids: List[str]):
    """Enhanced Neo4j context retrieval."""
    context = {
        "nodes": [], 
        "relationships": [], 
        "cities": set(), 
        "tags": set()
    }
    
    with driver.session() as session:
        for nid in node_ids:
            # Get node details
            node_q = """
            MATCH (n:Entity {id: $nid})
            RETURN n.id AS id, n.name AS name, n.type AS type,
                   n.description AS description, n.city AS city,
                   n.tags AS tags, n.best_time_to_visit AS best_time
            """
            for rec in session.run(node_q, nid=nid):
                node_data = dict(rec)
                context["nodes"].append(node_data)
                if rec["city"]:
                    context["cities"].add(rec["city"])
                if rec["tags"]:
                    context["tags"].update(rec["tags"])
            
            # Multi-hop relationships
            rel_q = """
            MATCH (n:Entity {id: $nid})-[r*1..2]-(m:Entity)
            WITH n, m LIMIT 15
            RETURN m.id AS id, m.name AS name, m.type AS type,
                   m.description AS desc, m.city AS city
            """
            for rec in session.run(rel_q, nid=nid):
                context["relationships"].append(dict(rec))
    
    return context

def expand_query(query: str) -> List[str]:
    """Generate query variations."""
    expansions = [query]
    keywords = {
        "romantic": ["couples", "honeymoon"],
        "adventure": ["trekking", "outdoor"],
        "beach": ["coastal", "resort"],
        "culture": ["heritage", "history"],
        "food": ["cuisine", "dining"]
    }
    for key, exps in keywords.items():
        if key in query.lower():
            expansions.append(f"{query} {exps[0]}")
            break
    return expansions[:2]

def build_prompt(query: str, matches: List, context: Dict):
    """Construct enhanced prompt."""
    system = """You are an expert Vietnam travel advisor providing detailed, practical recommendations.

Format your response with clear sections:
- Overview of recommendations
- Specific places with their IDs
- Day-by-day itinerary if applicable
- Practical travel tips

Keep responses comprehensive yet well-structured."""

    vec_ctx = "**Top Matching Locations:**\n\n"
    for i, m in enumerate(matches[:5], 1):
        meta = m["metadata"]
        vec_ctx += f"{i}. **{meta.get('name', 'Unknown')}** (ID: `{m['id']}`)\n"
        vec_ctx += f"   - Type: {meta.get('type')} | City: {meta.get('city')}\n"
        vec_ctx += f"   - Tags: {', '.join(meta.get('tags', []))}\n\n"
    
    graph_ctx = "**Related Places & Context:**\n\n"
    for node in context["nodes"][:5]:
        graph_ctx += f"- **{node['name']}** ({node['type']})\n"
        desc = node['description'][:150] if node['description'] else "No description"
        graph_ctx += f"  {desc}...\n\n"
    
    insights = f"**Coverage:** {len(context['cities'])} cities | Themes: {', '.join(list(context['tags'])[:5])}"
    
    user_msg = f"""Question: "{query}"

{vec_ctx}

{graph_ctx}

{insights}

Provide a comprehensive, well-structured response with specific recommendations."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ]

def get_response(query: str):
    """Generate response using hybrid RAG system."""
    start_time = time.time()
    
    # Query expansion
    expanded = expand_query(query)
    
    # Gather matches
    all_matches = []
    for exp_q in expanded:
        matches = pinecone_query(exp_q, top_k=5)
        all_matches.extend(matches)
    
    # Deduplicate
    seen = set()
    unique_matches = []
    for m in all_matches:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique_matches.append(m)
    matches = unique_matches[:5]
    
    # Get graph context
    ids = [m["id"] for m in matches]
    context = fetch_graph_context(ids)
    
    # Store stats
    stats = {
        "vector_results": len(matches),
        "graph_nodes": len(context["nodes"]),
        "graph_relationships": len(context["relationships"]),
        "cities": len(context["cities"]),
        "themes": list(context["tags"])[:5],
        "response_time": 0
    }
    
    # Build prompt and generate
    prompt = build_prompt(query, matches, context)
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            max_tokens=800,
            temperature=0.3
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Error generating response: {e}"
    
    stats["response_time"] = round(time.time() - start_time, 2)
    
    return answer, stats

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Vietnam Travel Assistant</h1>
    <p class="subtitle">Intelligent travel planning powered by AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: white;">
        <h2 style="color: white; margin: 0;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Select Page",
        ["Chat", "About", "Stats"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if page == "Chat":
        st.markdown("""
        <div style="color: white; padding: 1rem 0;">
            <h3 style="color: white;">Example Queries</h3>
        </div>
        """, unsafe_allow_html=True)
        
        example_queries = [
            "Romantic places in Vietnam",
            "4-day itinerary for Vietnam",
            "Adventure activities in Sapa",
            "Best beaches in Vietnam",
            "Cultural sites in Hanoi"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"ex_{query}", use_container_width=True):
                st.session_state.example_query = query
    
    st.markdown("---")
    
    # System info
    st.markdown("""
    <div style="color: white; padding: 1rem 0;">
        <h3 style="color: white;">System Info</h3>
        <div class="stat-box">
            <div class="stat-number">360</div>
            <div class="stat-label">Locations</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">10</div>
            <div class="stat-label">Cities</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content area
if page == "Chat":
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-label">You</div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-label">Assistant</div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Your question",
                placeholder="Ask about destinations, itineraries, activities...",
                key="user_input",
                label_visibility="collapsed"
            )
            submit = st.form_submit_button("Send", use_container_width=True)
        
        # Handle example query from sidebar
        if 'example_query' in st.session_state:
            user_input = st.session_state.example_query
            submit = True
            del st.session_state.example_query
        
        if submit and user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Show processing status
            with st.spinner("Searching knowledge base..."):
                response, stats = get_response(user_input)
                st.session_state.search_stats = stats
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
            st.rerun()
    
    with col2:
        if st.session_state.search_stats:
            st.markdown("### Last Search")
            stats = st.session_state.search_stats
            
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{stats['vector_results']}</div>
                <div class="stat-label">Vector Matches</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{stats['graph_nodes']}</div>
                <div class="stat-label">Graph Nodes</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{stats['cities']}</div>
                <div class="stat-label">Cities Covered</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{stats['response_time']}s</div>
                <div class="stat-label">Response Time</div>
            </div>
            """, unsafe_allow_html=True)
            
            if stats['themes']:
                st.markdown("**Themes:**")
                st.write(", ".join(stats['themes']))

elif page == "About":
    st.markdown("""
    <div class="chat-container">
        <h2 style="color: #0077BE;">About This System</h2>
        
        <p>This intelligent travel assistant uses a hybrid RAG (Retrieval-Augmented Generation) system combining:</p>
        
        <h3 style="color: #13A5B1;">Technologies</h3>
        <ul>
            <li><strong>Pinecone Vector Database</strong> - Semantic search for finding conceptually similar destinations</li>
            <li><strong>Neo4j Graph Database</strong> - Relationship mapping between cities, attractions, hotels, and activities</li>
            <li><strong>OpenAI GPT-4</strong> - Natural language understanding and response generation</li>
        </ul>
        
        <h3 style="color: #13A5B1;">Features</h3>
        <ul>
            <li>Multi-day itinerary planning</li>
            <li>Personalized recommendations based on preferences</li>
            <li>Relationship-aware suggestions (nearby hotels, connected cities)</li>
            <li>Semantic search across 360+ locations</li>
        </ul>
        
        <h3 style="color: #13A5B1;">Data Coverage</h3>
        <ul>
            <li>10 major Vietnamese cities</li>
            <li>150+ attractions</li>
            <li>100+ hotels</li>
            <li>100+ activities</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Stats":
    st.markdown("### System Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">360</div>
            <div class="stat-label">Total Entities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">700+</div>
            <div class="stat-label">Relationships</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">1536</div>
            <div class="stat-label">Vector Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.messages:
        st.markdown("### Conversation Statistics")
        st.write(f"Total messages: {len(st.session_state.messages)}")
        st.write(f"User queries: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")
        st.write(f"Assistant responses: {len([m for m in st.session_state.messages if m['role'] == 'assistant'])}")

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Vietnam Travel Assistant</strong> | Powered by Hybrid RAG Technology</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Pinecone × Neo4j × OpenAI</p>
</div>
""", unsafe_allow_html=True)
