import streamlit as st
from test import LegalRAGSystem
import time

st.set_page_config(
    page_title="Indian Legal Document Search",
    page_icon="⚖️",
    layout="wide"
)

@st.cache_resource(show_spinner=False)
def initialize_system(test_mode=True):
    """Initialize the RAG system with caching and error handling."""
    try:
        with st.spinner("Initializing legal AI system (this may take a few minutes)..."):
            start_time = time.time()
            rag_system = LegalRAGSystem()
            
            # Initialize models first
            rag_system.initialize_models()
            
            # Load data and generate/store embeddings if needed
            rag_system.load_data(test_mode=test_mode, test_size=1000)
            rag_system.generate_and_store_embeddings(test_mode=test_mode, test_size=1000)
            
            st.success(f"System initialized in {time.time() - start_time:.1f} seconds")
            return rag_system
            
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.stop()

def display_results(query, result, rag_system):
    """Display results with improved layout and document viewing."""
    st.success("Analysis complete!")
    
    # Main answer section
    with st.container():
        st.subheader("Legal Analysis")
        st.markdown(f"**Question:** {query}")
        
        # Improved answer display with expandable details
        with st.expander("📝 View Detailed Answer", expanded=True):
            st.markdown(result['answer'])
    
    # Supporting documents section
    st.subheader("Supporting Documents")
    cols = st.columns(2)  # Two-column layout for documents
    
    for i, doc in enumerate(result['documents']):
        # Alternate between columns for better space utilization
        col = cols[i % 2]
        
        with col:
            with st.expander(f"📄 {doc['title']} (Relevance: {doc['score']:.2f})"):
                st.caption(f"**Court:** {doc['court']} | **Type:** {doc['case_type']}")
                
                # Tabbed document view
                tab1, tab2 = st.tabs(["Preview", "Full Document"])
                
                with tab1:
                    st.markdown(f"**Excerpt:**\n{doc['text']}...")
                
                with tab2:
                    full_text = rag_system.get_full_text(doc['full_text_id'])
                    
                    # Scrollable document viewer
                    st.markdown("""
                    <div style="height: 300px; overflow-y: scroll; border: 1px solid #e1e4e8; 
                                padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    """, unsafe_allow_html=True)
                    
                    st.markdown(full_text)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Download button with better styling
                    st.download_button(
                        label="📥 Download Full Document",
                        data=full_text,
                        file_name=f"{doc['title']}.txt",
                        mime="text/plain",
                        key=f"download_{doc['full_text_id']}",
                        use_container_width=True
                    )

def main():
    st.title("⚖️ Indian Legal Document Search & Analysis")
    st.markdown("""
    <style>
        .stDownloadButton button {
            width: 100%;
        }
        .st-emotion-cache-1v0mbdj {
            max-width: 100%;
        }
        .st-emotion-cache-1pbsqtx {
            font-size: 1.1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with settings
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 3)
        max_length = st.slider("Max response length (tokens)", 100, 1000, 300)
        test_mode = st.toggle("Test Mode (faster)", True, 
                             help="Uses smaller dataset for quicker testing")
        
        st.markdown("---")
        st.markdown("**System Information**")
        st.markdown("""
        - **Embedding Storage:** ChromaDB (persistent)
        - **LLM:** Mistral-7B-Instruct
        - **Documents:** InJudgements dataset
        """)
        
        if st.button("Clear Cache", help="Clear system cache and restart"):
            st.cache_resource.clear()
            st.rerun()
    
    # Initialize system (with test mode toggle)
    rag_system = initialize_system(test_mode=test_mode)
    
    # Main query input
    query = st.text_input(
        "Enter your legal query:",
        placeholder="E.g., 'What are the provisions for land acquisition compensation?'",
        key="legal_query_input",
        label_visibility="collapsed"
    )
    
    # Search button with better visual feedback
    if st.button("Search Legal Documents", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a query")
            return
            
        with st.spinner("Analyzing legal documents..."):
            try:
                # Progress bar with status updates
                status = st.empty()
                progress_bar = st.progress(0)
                
                status.markdown("🔍 Retrieving relevant documents...")
                documents = rag_system.retrieve_documents(query, top_k=top_k)
                progress_bar.progress(50)
                
                status.markdown("🤖 Generating legal analysis...")
                answer = rag_system.generate_response(query, documents, max_new_tokens=max_length)
                progress_bar.progress(100)
                
                status.empty()
                progress_bar.empty()
                
                display_results(query, {
                    "answer": answer,
                    "documents": documents
                }, rag_system)
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()
