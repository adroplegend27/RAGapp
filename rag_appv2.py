import streamlit as st
import os
import json
import subprocess
import sys
from datetime import datetime
import openai  # Replace with deepseek client if using DeepSeek
from dotenv import load_dotenv
import sqlite3

# Load environment variables
load_dotenv()

# Configure API keys
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or DeepSeek/Claude key

# Set page configuration
st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Import your existing search functions from index.py
# This assumes your index.py is in the same directory and has these functions
sys.path.append(os.path.dirname(__file__))
try:
    from index import search_documents, search_timeline, find_contradictions, process_data_directory, close_connections
except ImportError as e:
    st.error(f"Error importing functions from index.py: {str(e)}")
    st.error("Make sure index.py is in the same directory and contains these functions.")
    st.stop()

# Database debugging function
def check_database_status():
    db_path = "data/database/timeline.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            st.sidebar.success(f"Database contains {count} documents")
            
            # Show the first 5 documents for verification
            cursor.execute("SELECT id, type, subject, filename FROM documents LIMIT 5")
            docs = cursor.fetchall()
            with st.sidebar.expander("Sample documents in database"):
                for doc in docs:
                    st.write(f"{doc[0]}: {doc[2] or doc[3]} ({doc[1]})")
            
            conn.close()
            return True
        except Exception as e:
            st.sidebar.error(f"Database error: {str(e)}")
            return False
    else:
        st.sidebar.error(f"Database file not found: {os.path.abspath(db_path)}")
        return False

# Main title
st.title("⚖️ Legal Document AI Assistant")

# Sidebar for file upload and indexing
with st.sidebar:
    st.header("Document Management")
    
    # Check database status
    st.subheader("Database Status")
    db_exists = check_database_status()
    
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload new documents", accept_multiple_files=True, type=["json", "pdf", "txt"])
    
    if uploaded_files:
        # Save uploaded files
        save_dir = "uploads"
        os.makedirs(save_dir, exist_ok=True)
        
        files_saved = []
        for file in uploaded_files:
            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            files_saved.append(file.name)
        
        if files_saved:
            st.success(f"Saved {len(files_saved)} files")
            
        if st.button("Process Uploaded Files"):
            # Direct import and execution instead of subprocess
            st.info("Processing files... This may take a moment.")
            with st.spinner("Indexing documents..."):
                try:
                    # Import the function directly
                    count = process_data_directory(save_dir)
                    if count > 0:
                        st.success(f"Successfully indexed {count} files")
                    else:
                        st.warning("No files were indexed. Check file format or content.")
                except Exception as e:
                    st.error(f"Error indexing files: {str(e)}")
    
    st.markdown("---")
    
    st.subheader("About")
    st.write("This tool helps you analyze legal documents using AI and search capabilities.")
    
    # Display paths for debugging
    with st.expander("Debug Information"):
        st.write(f"Working directory: {os.getcwd()}")
        st.write(f"Database path: {os.path.abspath('data/database/timeline.db')}")
        st.write(f"Index path: {os.path.abspath('data/indexes')}")
        
        # Check if index directory exists and has files
        index_path = "data/indexes"
        if os.path.exists(index_path) and os.path.isdir(index_path):
            index_files = os.listdir(index_path)
            st.write(f"Index directory contains {len(index_files)} files")
        else:
            st.error(f"Index directory not found or empty: {os.path.abspath(index_path)}")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["AI Assistant", "Search", "Timeline", "Contradictions"])

with tab1:
    st.header("AI Assistant")
    
    # User query input
    user_query = st.text_area("Ask a question about your documents:", height=100)
    
    if st.button("Get Answer") and user_query:
        # First check if database exists
        if not os.path.exists("data/database/timeline.db"):
            st.error("Database not found. Please index your documents first.")
            st.stop()
        
        with st.spinner("Generating response..."):
            # Parse the user query to extract keywords
            # This is a simple approach to make searching more context-aware
            import re
            from collections import Counter
    
            # Extract meaningful keywords from the query
            def extract_keywords(query):
                # Remove common words and punctuation
                stop_words = {"hello", "gpt", "what", "was", "is", "are", "the", "a", "an", "and", "or", 
                       "but", "in", "on", "at", "to", "for", "with", "about", "from", "of", "by",
                       "me", "my", "mine", "you", "your", "yours", "please", "tell", "show", "i", "we"}
        
                # Tokenize and clean
                words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        
                # Filter out stop words and short words
                keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
                # If no keywords found, fall back to basic query terms or use all
                if not keywords and len(words) > 0:
                    # Use the most significant terms from the original query
                    return words[-2:] if len(words) > 1 else words
        
                return keywords
    
            # Get keywords from the query
            keywords = extract_keywords(user_query)
    
            # Debug information - show search results
            with st.expander("Search Results"):
                st.write(f"Extracted keywords: {', '.join(keywords) if keywords else 'None found, using all documents'}")
        
                # If we have keywords, search using them
                if keywords:
                    # Create a search string using the keywords (OR logic to be more inclusive)
                    search_string = " OR ".join(keywords)
                    st.write(f"Search string: '{search_string}'")
                    search_results = search_documents(search_string, limit=10)
                else:
                    # If no good keywords, get the most recent documents instead
                    st.write("No specific keywords found. Retrieving most recent documents.")
                    # This assumes your index.py has a function to get recent documents
                    # If it doesn't, you'll need to add one or use a simple search like:
                    search_results = search_documents("email OR document OR receipt OR statement", limit=5)
        
                st.write(f"Found {len(search_results)} relevant documents")
        
                # If very few or no results, try a more general search 
                if len(search_results) < 2:
                    st.write("Few results found. Trying a more general search...")
                    backup_results = search_documents("email OR document OR attachment", limit=5)
                    st.write(f"Found {len(backup_results)} additional documents")
            
                    # Combine the results
                    search_results = search_results + [doc for doc in backup_results if doc not in search_results]
        
                if len(search_results) > 0:
                    for i, doc in enumerate(search_results, 1):
                        doc_type = "Email" if doc.get("type") == "email" else "Attachment"
                        title = doc.get("subject", doc.get("filename", f"Document {i}"))
                        st.write(f"Document {i}: {title} ({doc_type})")
                else:
                    st.error("No documents found in the index. This could mean:")
                    st.error("1. The documents haven't been indexed properly")
                    st.error("2. The index doesn't contain relevant information")
                    st.error("3. There might be an issue with the search functionality")
            
                    # Provide a fallback response
                    st.markdown("### Answer")
                    st.markdown("I'm sorry, but I couldn't find any relevant documents to answer your question. Please try:")
                    st.markdown("1. Making sure your documents are properly indexed")
                    st.markdown("2. Rephrasing your question")
                    st.markdown("3. Using the Search tab to verify if your documents are searchable")
                    st.stop()

            
            # Also check for potential contradictions
            contradictions = find_contradictions(user_query)
            
            # Format context for the AI
            context = "Relevant information from documents:\n\n"
            
            for i, doc in enumerate(search_results, 1):
                doc_type = "Email" if doc.get("type") == "email" else "Attachment"
                date = doc.get("date", "Unknown date")
                source = doc.get("from_field", "Unknown source") if doc_type == "Email" else doc.get("filename", "Unknown file")
                content = doc.get("content", "")[:1000]  # Limit content length
                
                context += f"{i}. {doc_type} from {source} on {date}:\n{content}\n\n"
            
            if contradictions:
                context += "Potential contradictions detected:\n\n"
                for i, contradiction in enumerate(contradictions, 1):
                    context += f"Contradiction {i}: From {contradiction['source']}\n"
                    context += f"- Initially stated ({contradiction['statement1']['date']}): {contradiction['statement1']['content']}\n"
                    context += f"- Later stated ({contradiction['statement2']['date']}): {contradiction['statement2']['content']}\n\n"
            
            # Send to AI with appropriate prompting
            try:
                # Using OpenAI's GPT-4. Replace with DeepSeek or Claude model if needed
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # or DeepSeek/Claude model
                    messages=[
                        {"role": "system", "content": f"You are a helpful legal assistant that analyzes documents. Today is {datetime.now().strftime('%Y-%m-%d')}. Base your answers only on the provided document information and highlight contradictions if present. If the user asks a question that can't be answered with the provided documents, explain what information is missing."},
                        {"role": "user", "content": f"I asked: '{user_query}'\n\nHere is the relevant information from the documents:\n\n{context}\n\nPlease answer my question based only on these documents. If the documents don't provide enough information to answer fully, explain what's missing."}
                    ],
                    temperature=0.1  # Low temperature for more factual responses
                )

                
                # Display response
                st.markdown("### Answer")
                st.markdown(response.choices[0].message["content"])
                
                # Display sources
                st.markdown("### Sources")
                for i, doc in enumerate(search_results, 1):
                    doc_type = "Email" if doc.get("type") == "email" else "Attachment"
                    date = doc.get("date", "Unknown date")
                    source = doc.get("from_field", "Unknown source") if doc_type == "Email" else doc.get("filename", "Unknown file")
                    
                    with st.expander(f"{i}. {doc_type} from {source} on {date}"):
                        st.text(doc.get("content", "")[:2000] + "..." if len(doc.get("content", "")) > 2000 else doc.get("content", ""))
                
            except Exception as e:
                st.error(f"Error generating AI response: {str(e)}")
                st.error("Make sure your OpenAI API key is correctly set in the .env file")

with tab2:
    st.header("Document Search")
    
    search_query = st.text_input("Search documents:")
    limit = st.slider("Maximum results", min_value=5, max_value=50, value=10)
    
    if st.button("Search") and search_query:
        with st.spinner("Searching..."):
            results = search_documents(search_query, limit=limit)
            
            if not results:
                st.warning("No results found.")
            else:
                st.success(f"Found {len(results)} documents")
                
                for i, result in enumerate(results, 1):
                    doc_type = "Email" if result.get("type") == "email" else "Attachment"
                    title = result.get("subject", result.get("filename", f"Document {i}"))
                    date = result.get("date", "Unknown date")
                    source = result.get("from_field", "Unknown source") if doc_type == "Email" else ""
                    
                    with st.expander(f"{i}. {title} ({doc_type}, {date})"):
                        if source:
                            st.write(f"**From:** {source}")
                        st.write("**Content:**")
                        st.text(result.get("content", "")[:1000] + "..." if len(result.get("content", "")) > 1000 else result.get("content", ""))

with tab3:
    st.header("Timeline View")
    
    timeline_query = st.text_input("Search for timeline:")
    timeline_limit = st.slider("Maximum timeline results", min_value=5, max_value=30, value=10, key="timeline_slider")
    
    if st.button("Generate Timeline") and timeline_query:
        with st.spinner("Building timeline..."):
            results = search_timeline(timeline_query, limit=timeline_limit)
            
            if not results:
                st.warning("No timeline data found.")
            else:
                st.success(f"Found {len(results)} timeline entries")
                
                # Create a simple timeline visualization
                for i, doc in enumerate(results, 1):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.write(f"**{doc['date']}**")
                        st.write(f"*{doc['from']}*")
                    
                    with col2:
                        st.write(f"**{doc['subject'] or doc['filename']}**")
                        st.write(doc['content_snippet'])
                        
                        for event in doc['timeline']:
                            st.info(f"{event['date']}: {event['type']} - {event['details']}")
                    
                    st.divider()

with tab4:
    st.header("Contradiction Detector")
    
    topic = st.text_input("Topic to check for contradictions:")
    
    if st.button("Find Contradictions") and topic:
        with st.spinner("Analyzing contradictions..."):
            results = find_contradictions(topic)
            
            if not results:
                st.success("No contradictions found on this topic.")
            else:
                st.warning(f"Found {len(results)} potential contradictions")
                
                for i, contradiction in enumerate(results, 1):
                    st.subheader(f"Contradiction {i}: Statements from {contradiction['source']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Initial Statement ({contradiction['statement1']['date']}):**")
                        st.info(contradiction['statement1']['content'])
                    
                    with col2:
                        st.markdown(f"**Later Statement ({contradiction['statement2']['date']}):**")
                        st.error(contradiction['statement2']['content'])
                    
                    st.divider()

# This function would try to close connections, but since Streamlit is stateless
# it's better to make our database operations self-contained
def on_shutdown():
    try:
        close_connections()
    except:
        pass

# Register the shutdown handler
# Note: This may not work in all environments due to Streamlit's execution model
import atexit
atexit.register(on_shutdown)
