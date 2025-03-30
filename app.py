import streamlit as st
from backend import LegalRAGSystem
from huggingface_hub import login
import os
import time

login(token=st.secrets["HUGGINGFACE_TOKEN"])

st.set_page_config(
    page_title="Indian Legal Document Search",
    page_icon="⚖️",
    layout="wide"
)

def get_huggingface_token():
    """Get HF token from secrets or environment variables."""
    try:
        # Try Streamlit secrets first
        return st.secrets["HUGGINGFACE_TOKEN"]
    except (KeyError, AttributeError):
        # Fallback to environment variable
        return os.getenv("HUGGINGFACE_TOKEN")

def validate_environment():
    """Check for required environment variables."""
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_TOKEN"):
        st.error("""
        🔑 Authentication required. Please configure:
        1. For local development: Set HF_TOKEN environment variable
        2. For Streamlit Cloud: Add to Settings → Secrets as HF_TOKEN
        """)
        st.stop()

def check_environment():
    """Validate required environment variables."""
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_TOKEN"):
        st.error("""
        🔐 Authentication Required
        
        Please set up your Hugging Face token:
        1. For local dev: Run `export HF_TOKEN=your_token`
        2. For Streamlit: Add to Settings → Secrets
        """)
        st.stop()

@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize system with deployment-friendly settings."""
    check_environment()
    
    try:
        with st.spinner("🔄 Initializing system (may take 2-3 minutes)..."):
            rag_system = LegalRAGSystem()
            
            # Step-by-step initialization with progress
            steps = [
                ("Loading legal documents", rag_system.load_data),
                ("Initializing AI models", rag_system.initialize_models),
                ("Processing documents", rag_system.generate_embeddings)
            ]
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            for i, (message, func) in enumerate(steps):
                status.markdown(f"**Step {i+1}/{len(steps)}:** {message}...")
                func()
                progress_bar.progress((i + 1) / len(steps))
            
            status.success("✅ System ready!")
            time.sleep(1)
            return rag_system
            
    except Exception as e:
        st.error(f"""
        🛑 Initialization failed
        
        Error: {str(e)}
        
        Solutions:
        1. Check your Hugging Face token
        2. Verify internet connection
        3. Try again later if servers are busy
        """)
        st.stop()

def display_results(query, result, rag_system):
    """Display results with full document viewing options."""
    st.success("Analysis complete!")
    
    st.subheader("Legal Analysis")
    st.markdown(f"**Question:** {query}")
    st.markdown(f"**Answer:** {result['answer']}")
    
    st.subheader("Supporting Documents")
    for i, doc in enumerate(result['documents']):
        with st.expander(f"📄 Document {i+1}: {doc['title']} (Relevance: {doc['score']:.2f})"):
            st.markdown(f"""
            **Court:** {doc['court']}  
            **Type:** {doc['case_type']}  
            """)
            
            tab1, tab2 = st.tabs(["Preview", "Full Document"])
            
            with tab1:
                st.markdown(f"**Excerpt:**\n{doc['text']}...")
            
            with tab2:
                full_text = rag_system.get_full_text(doc['full_text_id'])
                
                st.markdown("""
                <div style="height: 300px; overflow-y: scroll; border: 1px solid #e1e4e8; 
                            padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                """, unsafe_allow_html=True)
                
                st.markdown(full_text)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Full Document",
                    data=full_text,
                    file_name=f"{doc['title']}.txt",
                    mime="text/plain",
                    key=f"download_{doc['full_text_id']}"
                )

def main():
    st.title("⚖️ Indian Legal Document Search & Analysis")
    
    rag_system = initialize_system()
    
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of documents to retrieve", 1, 5, 3)
        max_length = st.slider("Max response length", 100, 500, 300)
    
    query = st.text_input(
        "Enter your legal query:",
        placeholder="E.g., 'What are the provisions for land acquisition compensation?'",
        key="legal_query_input"
    )
    
    if st.button("Search Legal Documents", type="primary", key="search_button"):
        if not query:
            st.warning("Please enter a query")
            return
            
        with st.spinner("Analyzing legal documents..."):
            try:
                progress_bar = st.progress(0)
                documents = rag_system.retrieve_documents(query, top_k=top_k)
                progress_bar.progress(50)
                
                answer = rag_system.generate_response(query, documents, max_new_tokens=max_length)
                progress_bar.progress(100)
                
                display_results(query, {
                    "answer": answer,
                    "documents": documents
                }, rag_system)
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()