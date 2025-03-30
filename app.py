import streamlit as st
from backend import LegalRAGSystem
import time
import base64

# Configure Streamlit
st.set_page_config(
    page_title="Indian Legal Document Search",
    page_icon="⚖️",
    layout="wide"
)

@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize the RAG system with proper error handling."""
    try:
        with st.spinner("Initializing legal AI system..."):
            rag_system = LegalRAGSystem()
            
            # Step 1: Load data
            rag_system.df = rag_system.load_data()
            
            # Step 2: Initialize models (embedding + LLM)
            rag_system.initialize_models()
            
            # Step 3: Generate embeddings
            rag_system.generate_embeddings()
            
            return rag_system
            
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
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
            
            # Add tabs for preview and full document
            tab1, tab2 = st.tabs(["Preview", "Full Document"])
            
            with tab1:
                st.markdown(f"**Excerpt:**\n{doc['text']}...")
            
            with tab2:
                full_text = rag_system.get_full_text(doc['full_text_id'])
                
                # Create a scrollable container for the full text
                st.markdown("""
                <div style="height: 300px; overflow-y: scroll; border: 1px solid #e1e4e8; 
                            padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                """, unsafe_allow_html=True)
                
                st.markdown(full_text)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add download button with unique key
                st.download_button(
                    label="Download Full Document",
                    data=full_text,
                    file_name=f"{doc['title']}.txt",
                    mime="text/plain",
                    key=f"download_{doc['full_text_id']}"  # Unique key for each button
                )

def main():
    st.title("⚖️ Indian Legal Document Search & Analysis")
    
    # Initialize system
    rag_system = initialize_system()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of documents to retrieve", 1, 5, 3)
        max_length = st.slider("Max response length", 100, 500, 300)
    
    # Main interface
    query = st.text_input(
        "Enter your legal query:",
        placeholder="E.g., 'What are the provisions for land acquisition compensation?'",
        key="legal_query_input"  # Unique key for input
    )
    
    if st.button("Search Legal Documents", type="primary", key="search_button"):
        if not query:
            st.warning("Please enter a query")
            return
            
        with st.spinner("Analyzing legal documents..."):
            try:
                # Step 1: Document retrieval
                progress_bar = st.progress(0)
                documents = rag_system.retrieve_documents(query, top_k=top_k)
                progress_bar.progress(50)
                
                # Step 2: Generate answer
                answer = rag_system.generate_response(query, documents, max_new_tokens=max_length)
                progress_bar.progress(100)
                
                # Display results - now passing rag_system for full text access
                display_results(query, {
                    "answer": answer,
                    "documents": documents
                }, rag_system)
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()