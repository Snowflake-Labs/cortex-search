import streamlit as st # Import python packages
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
from snowflake.core import Root
import os

# LangChain imports for structured prompt and tool management
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool

import pandas as pd
import json

pd.set_option("max_colwidth",None)

# service parameters
SOURCE_DOCS_STAGE = "@CORTEX_SEARCH_DOCS.DATA.DOCS"
SOURCE_DOCS_PATH = "raw_pdf"
CORTEX_SEARCH_DATABASE = "CORTEX_SEARCH_DOCS"
CORTEX_SEARCH_SCHEMA = "DATA"
CORTEX_SEARCH_SERVICE = "SEARCH_SERVICE_MULTIMODAL"
GRAPH_DB = "DOCS_EDGES"

### Default Values
DEFAULT_NUM_CHUNKS = 3
DEFAULT_NUM_HOPS = 3
CHUNK_OPTIONS = [1, 2, 3, 4, 5]

NUM_CHUNKS = st.session_state.get("num_chunks", DEFAULT_NUM_CHUNKS)
NUM_HOPS = st.session_state.get("num_hops", DEFAULT_NUM_HOPS)
SLIDE_WINDOW = 5 # how many last conversations to remember

# Define columns for the multimodal service
COLUMNS = [
    "text",
    "page_number", 
    "image_filepath"
]

def init_session():
    """Initialize the Snowflake session"""
    try:
        # Try to get active session (works in SiS)
        return get_active_session()
    except:
        # For local development, use st.connection
        conn = st.connection("snowflake", type="snowflake")
        return conn.session()

session = init_session()
root = Root(session)
svc = None

# Initialize the service based on the selected service
def init_service():
    global svc  # Make svc a global variable
    svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]
    return svc

# Initialize the service
init_service()

@tool
def search_similar_documents(query: str) -> list:
    """
    Search for documents similar to the given query using multimodal search.
    
    Uses Snowflake's multimodal embedding model to find relevant documents
    based on both text and image content.
    
    Args:
        query: The search query text to find similar documents
        
    Returns:
        List of similar documents with their metadata including image paths
    """
    # Embed the query using the same multimodal model used for image embeddings
    sql_output = session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_1024('voyage-multimodal-3',
        '{query.replace("'", "")}')
    """).collect()
    query_vector = list(sql_output[0].asDict().values())[0]

    response = svc.search(query, COLUMNS, limit=NUM_CHUNKS,
                            experimental={'queryEmbedding': query_vector})

    if st.session_state.get('debug', False):
        page_numbers = [doc.get('page_number', '') for doc in response.results if doc.get('page_number')]
        st.sidebar.write(f" 📄 Found {len(response.results)} similar docs: {page_numbers}")
    
    return response.results

@tool
def search_connected_documents(source_paths: list, num_hops: int = None) -> list:
    """
    Search for documents that are logically connected to the given source documents.
    
    Uses Snowflake's graph traversal capabilities to find related documents
    through citations, references, and other logical connections.
    
    Args:
        source_paths: List of document paths from search_similar_documents
        num_hops: Number of connection hops to traverse (default: user setting)
        
    Returns:
        List of connected documents with explanations of their relationships
    """
    if not source_paths:
        if st.session_state.get('debug', False):
            st.sidebar.write(f" 🔗 No source paths to search")
        return []
    
    if num_hops is None:
        num_hops = st.session_state.get("num_hops", DEFAULT_NUM_HOPS)
        
    # Build the recursive query to find connected images
    query = f"""
    SELECT
        DEST_PATH AS image_filepath,
        DEST_PAGE AS page_number,
        ARRAY_AGG(exp.value) AS explanations
    FROM TABLE(FIND_CONNECTED_PAGES(
            ARRAY_CONSTRUCT({','.join([f"'{path}'" for path in source_paths])}),
            {num_hops}
        )),
    LATERAL FLATTEN(input => EXPLANATIONS) exp
    GROUP BY 1, 2
    """

    try:
        results = session.sql(query).collect()
        # Convert Row objects to dictionaries with lowercase keys for consistent handling
        results_dict = [{k.lower(): v for k, v in doc.asDict().items()} for doc in results]

        if st.session_state.get('debug', False):
            if results_dict:
                page_numbers = [doc.get('page_number', '') for doc in results_dict if doc.get('page_number')]
                st.sidebar.write(f" 🔗 Found {len(results_dict)} connected docs: {page_numbers}")
            else:
                st.sidebar.write(f" 🔗 No connected documents found")
        return results_dict
    except Exception as e:
        st.error(f"Error querying connected documents: {str(e)}")
        return []

@tool
def summarize_chat_history(chat_history: list, current_question: str) -> str:
    """
    Summarize the chat history with the current question to create a better search query.
    
    Creates a contextual search query that incorporates previous conversation
    context to improve document retrieval relevance.
    
    Args:
        chat_history: List of previous chat messages
        current_question: The current user question
        
    Returns:
        A summarized query that incorporates chat history context
    """
    if not chat_history:
        return current_question
        
    # Format chat history for prompt
    history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
    
    prompt = f"""Based on the chat history and the question, generate a query that extends the question
    with the chat history provided. The query should be in natural language. 
    Answer with only the query. Do not add any explanation.
    
    Chat history: {history_str}
    
    Question: {current_question}"""
    
    # Use native Cortex Complete
    summary = Complete(st.session_state.model_name, prompt)
    
    if st.session_state.get('debug', False):
        st.sidebar.write("🔍 Enhanced query with chat history")
        st.sidebar.caption(summary)

    return summary.replace("'", "")

# Prompt template for consistent multimodal responses
@st.cache_resource
def get_response_prompt_template():
    """Get the prompt template for generating final responses"""
    return PromptTemplate.from_template("""
You are an expert technical assistant. Answer the user's question using the provided documents.
Provide a comprehensive, step-by-step answer based on the document content and context.
Be thorough and include all necessary details and step-by-step instructions.
Never reference page numbers or document sections - include the actual content.

DOCUMENT CONTEXT:
The following describes what each document covers:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION: {question}

DOCUMENT IMAGES TO ANALYZE:
Please analyze the following document images: {image_refs}

ANSWER USER QUESTION:
""")

# Main declarative workflow
def execute_declarative_workflow(question: str, chat_history: list = None) -> tuple:
    """
    Execute a clean, predictable RAG workflow using structured tools.
    
    This declarative approach follows a proven optimal sequence:
    1. Enhance query with chat history (if available)
    2. Search for similar documents using multimodal search
    3. Find connected documents through graph traversal
    4. Generate comprehensive multimodal response
    
    Args:
        question: User's question
        chat_history: Previous conversation context
        
    Returns:
        Tuple of (response_text, document_paths)
    """
    
    if st.session_state.get('debug', False):
        st.sidebar.write("📋 **Reasoning RAG Started**")
    
    try:
        # Enhance query with chat history if available
        if chat_history:
            enhanced_query = summarize_chat_history.invoke({
                "chat_history": chat_history,
                "current_question": question
            })
        else:
            enhanced_query = question
        
        # Search for similar documents
        st.sidebar.write(f"✅ Step 1: Semantic similarity search")
        similar_docs = search_similar_documents.invoke({"query": enhanced_query})

        
        # Search for connected documents
        st.sidebar.write(f"✅ Step 2: Relevancy graph search")
        similar_paths = [doc.get('image_filepath', '') for doc in similar_docs if doc.get('image_filepath')]
        
        connected_docs = [] 
        if similar_paths:
            connected_docs = search_connected_documents.invoke({
                "source_paths": similar_paths,
                "num_hops": st.session_state.get("num_hops", DEFAULT_NUM_HOPS)
            })
        
        # Combine all unique document paths
        all_paths = []
        connected_paths = [doc.get('image_filepath', '') for doc in connected_docs if doc.get('image_filepath')]
        all_paths.extend(connected_paths)
        
        # Add similar paths that aren't already included
        for path in similar_paths:
            if path and path not in all_paths:
                all_paths.append(path)
        
        # Create context information
        explanations = []
        for i, doc in enumerate(connected_docs):
            try:
                doc_explanations = json.loads(doc.get('explanations', '[]') or '[]')
            except (json.JSONDecodeError, TypeError):
                doc_explanations = doc.get('explanations', []) if isinstance(doc.get('explanations'), list) else []
            
            if doc_explanations:
                explanation_text = "\n".join([f"- {exp}" for exp in doc_explanations])
                doc_path = doc.get('image_filepath', '') or f"connected_doc_{i+1}"
                doc_name = doc_path.split('/')[-1] if '/' in doc_path else doc_path
                explanations.append(f"Document '{doc_name}' covers:\n{explanation_text}")
        
        context_info = "\n\n".join(explanations) if explanations else "No specific context available."
        
        st.sidebar.write(f"✅ Step 3: Generating response with {len(all_paths)} docs")
        
        # Step 6: Generate final multimodal response
        if all_paths:
            return generate_multimodal_response(question, all_paths, context_info, chat_history)
        else:
            return "I couldn't find any relevant documents for your question.", []
            
    except Exception as e:
        st.error(f"Workflow error: {str(e)}")
        return "I encountered an error while processing your request.", []

def generate_multimodal_response(question: str, document_paths: list, context: str, chat_history: list = None) -> tuple:
    """
    Generate final response using Snowflake's native multimodal capabilities.
    
    Uses structured prompt template for consistency and Snowflake's PROMPT() function
    with TO_FILE() for full multimodal support.
    
    Args:
        question: User's question
        document_paths: List of document paths to include
        context: Context information from connected documents
        chat_history: Previous conversation context
        
    Returns:
        Tuple of (response_text, document_paths)
    """
    
    # Format chat history
    history_str = ""
    if chat_history:
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
    
    # Create image file references for PROMPT function
    image_files = []
    for path in document_paths:
        image_files.append(f"TO_FILE('{SOURCE_DOCS_STAGE}', '{path}')")
    image_files_str = ",\n".join(image_files)
    
    # Create positional references for images in the prompt
    image_refs = " ".join([f"{{{i}}}" for i in range(len(document_paths))])
    
    # Use structured prompt template with positional references
    prompt_template = get_response_prompt_template()
    formatted_prompt = prompt_template.format(
        context=context,
        chat_history=history_str if history_str else "No previous conversation",
        question=question,
        image_refs=image_refs
    )
    
    # Escape single quotes in the prompt for SQL
    escaped_prompt = formatted_prompt.replace("'", "''")

    # Use Snowflake's native multimodal PROMPT function
    query = f"""
        SELECT AI_COMPLETE('{st.session_state.model_name}',
            PROMPT('{escaped_prompt}',
            {image_files_str})
        );
    """
            
    sql_output = session.sql(query).collect()
    response = list(sql_output[0].asDict().values())[0]

    return response, document_paths

### UI Configuration Functions

def config_options():
    st.sidebar.selectbox(
        'Select your model:',
        (
         'claude-3-7-sonnet', 'claude-3-5-sonnet', 'claude-4-opus', 'claude-4-sonnet',
         'openai-gpt-4.1', 'openai-o4-mini',
         'llama4-maverick', 'llama4-scout', 'pixtral-large'
        ), key="model_name")

    # Create two columns for the selectboxes to be side by side with equal width
    col1, col2 = st.sidebar.columns([1, 1])
    
    with col1:
        st.selectbox('Initial docs:',
                    CHUNK_OPTIONS, key="num_chunks",
                    index=CHUNK_OPTIONS.index(NUM_CHUNKS),
                    help="Number of documents to retrieve in **initial semantic similarity search**")
    
    with col2:
        st.selectbox('Hops:',
                   CHUNK_OPTIONS, key="num_hops",
                   index=CHUNK_OPTIONS.index(NUM_HOPS),
                   help="Number of connection hops to traverse in **secondary relevancy graph search**")

    st.sidebar.checkbox('Chat history', key="use_chat_history", value = False)
    st.sidebar.checkbox('Debug mode', key="debug", value = True)
    
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)
    st.sidebar.expander("Session State").write(st.session_state)

def init_messages():
    # Initialize chat history
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def get_chat_history():
    """Get the history from the st.session_state.messages according to the slide window parameter"""
    chat_history = []
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    for i in range(start_index, len(st.session_state.messages) - 1):
         chat_history.append(st.session_state.messages[i])
    return chat_history

def main():
    st.title(f":robot_face: :mag_right: Multi-hop RAG Assistant")
    st.write("Predictable, optimized multi-hop retrieval with full multimodal support")
    st.write("List of pre-processed documents that will be used to answer your questions:")
    docs_available = session.sql(f"ls {SOURCE_DOCS_STAGE}/{SOURCE_DOCS_PATH}").collect()
    list_docs = [doc["name"] for doc in docs_available]
    for doc in list_docs:
        st.markdown(f"- {doc}")
        
    config_options()
    init_messages()
     
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input("What do you want to know about your products?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
    
            question = question.replace("'","")
    
            with st.spinner(f"Multimodal RAG + {st.session_state.model_name} thinking..."):
                # Get chat history for context
                chat_history = get_chat_history() if st.session_state.use_chat_history else []
                
                # Execute the declarative RAG workflow
                response, relative_paths = execute_declarative_workflow(question, chat_history)
                
                response = response.replace("'", "")
                message_placeholder.markdown(response)

                # Display images inline
                if relative_paths:
                    st.write("**Related Documents:**")
                    cols = st.columns(3)  # Create 3 columns for images
                    for i, path in enumerate(relative_paths):
                        cmd2 = f"select GET_PRESIGNED_URL('{SOURCE_DOCS_STAGE}', '{path}', 360) as URL_LINK;"
                        df_url_link = session.sql(cmd2).to_pandas()
                        url_link = df_url_link._get_value(0,'URL_LINK')
                        
                        # Display image in the appropriate column
                        with cols[i % 3]:
                            st.image(url_link, caption=f"Document {i+1}", use_container_width=True)
                            st.markdown(f"**Source:** [{path}]({url_link})")

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 