import streamlit as st # Import python packages
from snowflake.snowpark.context import get_active_session

from snowflake.cortex import Complete
from snowflake.core import Root

import pandas as pd
import json

pd.set_option("max_colwidth",None)

# Cortex Search Service and Source Docs Stage parameters
CORTEX_SEARCH_DATABASE = "CORTEX_SEARCH_DB"
CORTEX_SEARCH_SCHEMA = "PYU"
CORTEX_SEARCH_SERVICE = "DEMO_SEC_CORTEX_SEARCH_SERVICE"
SOURCE_DOCS_STAGE = "@CORTEX_SEARCH_DB.PYU.MULTIMODAL_DEMO_INTERNAL"
SOURCE_DOCS_PATH = "raw_pdf"
######

### Default Values
# Default value for NUM_CHUNKS
DEFAULT_NUM_CHUNKS = 3
# Available options for NUM_CHUNKS
CHUNK_OPTIONS = [1, 2, 3, 4, 5]

NUM_CHUNKS = st.session_state.get("num_chunks", DEFAULT_NUM_CHUNKS)
SLIDE_WINDOW = 5 # how many last conversations to remember. This is the slide window.

# Define columns for the multimodal service
COLUMNS = [
    "text",
    "page_number",
    "image_filepath"
]

session = get_active_session()
root = Root(session)
svc = None

# Initialize the service based on the selected service
def init_service():
    global svc  # Make svc a global variable
    svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]
    return svc

# Initialize the service
init_service()

### Functions
     
def config_options():

    st.sidebar.selectbox('Select your model:',
                        ('claude-3-5-sonnet', 'pixtral-large'),
                        key="model_name")

    st.sidebar.selectbox('Select number of docs to retrieve:',
                        CHUNK_OPTIONS, key="num_chunks",
                        index=CHUNK_OPTIONS.index(NUM_CHUNKS))

    st.sidebar.checkbox('Chat history', key="use_chat_history", value = True)
    st.sidebar.checkbox('Debug mode', key="debug", value = True)
    
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)
    st.sidebar.expander("Session State").write(st.session_state)

def init_messages():

    # Initialize chat history
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def get_similar_chunks_search_service(query):
    
    # Embed the query using the same multimodal model used for image embeddings
    sql_output = session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_1024('voyage-multimodal-3',
        '{query.replace("'", "")}')
    """).collect()
    query_vector = list(sql_output[0].asDict().values())[0]

    response = svc.search(query, COLUMNS, limit=NUM_CHUNKS,
                            experimental={'queryEmbedding': query_vector})

    st.sidebar.json(response.json())
    
    return response.json()

def get_chat_history():
#Get the history from the st.session_stage.messages according to the slide window parameter
    
    chat_history = []
    
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    for i in range (start_index , len(st.session_state.messages) -1):
         chat_history.append(st.session_state.messages[i])

    return chat_history

def summarize_question_with_history(chat_history, question):
# To get the right context, use the LLM to first summarize the previous conversation
# This will be used to get embeddings and find similar chunks in the docs for context

    prompt = f"""
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natual language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
    
    sumary = Complete(st.session_state.model_name, prompt)   

    if st.session_state.debug:
        st.sidebar.text("Summary to be used to find similar chunks in the docs:")
        st.sidebar.caption(sumary)

    sumary = sumary.replace("'", "")

    return sumary

def create_prompt_multimodal(myquestion):
    # Get chat history and question summary
    chat_history = get_chat_history() if st.session_state.use_chat_history else []
    question_summary = summarize_question_with_history(chat_history, myquestion) if chat_history else myquestion
    
    # Get prompt context
    prompt_context = get_similar_chunks_search_service(question_summary)

    # Convert chat history to string format
    chat_history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history]) if chat_history else ""

    # Generate images placeholder string for the prompt
    if NUM_CHUNKS == 1:
        image_placeholders_str = "{0}"
    else:
        image_placeholders = [f"{{{i}}}" for i in range(NUM_CHUNKS)]
        image_placeholders_str = ", ".join(image_placeholders[:-1]) + f", and {image_placeholders[-1]}"

    prompt = f"""
           You are an expert AI assistant that extracts information from the document image(s) {image_placeholders_str}.
           You are specialized in accurately extracting screenshots, diagrams and structured data
           from tables presented within images, paying close attention to merged cells in tables.
           
           You also offer a chat experience considering the information included in the CHAT HISTORY
           provided between <chat_history> and </chat_history> tags..
          
           When answering the question contained between <question> and </question> tags
           be concise and do not hallucinate. If you don´t have the information just say so.

           Do not mention the IMAGES used in your answer, but do reference the content including any diagrams or text.
           Do not mention the CHAT HISTORY used in your answer.

           Only answer the question if you can extract it from the IMAGES provided.
           
           <chat_history>
           {chat_history_str}
           </chat_history>
           <question>  
           {myquestion}
           </question>
           Answer: 
           """

    json_data = json.loads(prompt_context)

    relative_paths = [item['image_filepath'] for item in json_data['results']]

    return prompt, relative_paths


def answer_question(myquestion):
    prompt, relative_paths = create_prompt_multimodal(myquestion)

    image_files = []
    for path in relative_paths:
        image_files.append(f"TO_FILE('{SOURCE_DOCS_STAGE}', '{path}')")
    image_files_str = ",\n".join(image_files)

    query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE('claude-3-5-sonnet',
            PROMPT('{prompt}',
            {image_files_str})
        );
    """
            
    sql_output = session.sql(query).collect()

    response = list(sql_output[0].asDict().values())[0]

    return response, relative_paths

def main():
    
    st.title(f":robot_face: :mag_right: Multimodal RAG Assistant")
    st.write("This is the list of documents you already have and that will be used to answer your questions:")
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
    
            with st.spinner(f"{st.session_state.model_name} thinking..."):
                response, relative_paths = answer_question(question)            
                response = response.replace("'", "")
                message_placeholder.markdown(response)

                # Display images inline
                if relative_paths:
                    st.write("Related Images:")
                    cols = st.columns(3)  # Create 3 columns for images
                    for i, path in enumerate(relative_paths):
                        cmd2 = f"select GET_PRESIGNED_URL('{SOURCE_DOCS_STAGE}', '{path}', 360) as URL_LINK;"
                        df_url_link = session.sql(cmd2).to_pandas()
                        url_link = df_url_link._get_value(0,'URL_LINK')
                        
                        # Display image in the appropriate column
                        with cols[i % 3]:
                            st.image(url_link, caption=f"Image {i+1}", use_container_width=True)
                            st.markdown(f"Doc: [{path}]({url_link})")

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()