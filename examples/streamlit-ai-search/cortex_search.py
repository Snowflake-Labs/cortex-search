import streamlit as st # Import python packages
from snowflake.core import Root

MODELS = [
    "mistral-large",
    "snowflake-arctic",
    "llama3-70b",
    "llama3-8b",
]

@st.cache_resource
def make_session():
    """
    Initialize the connection to snowflake using the `conn` connection stored in `.streamlit/secrets.toml`
    """
    conn = st.connection("conn", type="snowflake")
    return conn.session()

def get_available_search_services():
    """
    Returns list of cortex search services available in the current schema 
    """
    search_service_results = session.sql(f"SHOW CORTEX SEARCH SERVICES IN SCHEMA {db}.{schema}").collect()
    return [svc.name for svc in search_service_results]
    
def get_search_column(svc):
    """
    Returns the name of the search column for the provided cortex search service
    """
    search_service_result = session.sql(f"DESC CORTEX SEARCH SERVICE {svc}").collect()[0]
    return search_service_result.search_column

def init_layout():
    st.title("Cortex AI Search and Summary")
    st.sidebar.markdown(f"Current database and schema: `{db}.{schema}`".replace('"', ''))

def init_config_options():
    """
    Initialize sidebar configuration options
    """
    st.text_area("Search:", value="", key="query", height=100)
    st.sidebar.selectbox("Cortex Search Service", get_available_search_services(), key="cortex_search_service")
    st.sidebar.number_input("Results", value=5, key="limit", min_value=3, max_value=10)
    st.sidebar.selectbox("Summarization model", MODELS, key="model")
    st.sidebar.toggle("Summarize", key="summarize", value = False)

def query_cortex_search_service(query):
    """
    Queries the cortex search service in the session state and returns a list of results
    """
    cortex_search_service = (
        root
        .databases[db]
        .schemas[schema]
        .cortex_search_services[st.session_state.cortex_search_service]
    )
    context_documents = cortex_search_service.search(query, [], limit=st.session_state.limit)
    return context_documents.results

def complete(model, prompt):
    """
    Queries the cortex COMPLETE LLM function with the provided model and prompt
    """
    try:
        resp = session.sql("select snowflake.cortex.complete(?, ?)", params=(model, prompt)).collect()[0][0]
    except Exception as e:
        resp = f"COMPLETE error: {e}"
    return resp

def summarize_search_results(results, query, search_col):
    """
    Returns an AI summary of the search results based on the user's query
    """
    search_result_str = ""
    for i, r in enumerate(results):
        search_result_str += f"Result {i+1}: {r[search_col]} \n"

    prompt = f"""
        [INST]
        You are a helpful AI Assistant embedded in a search application. You will be provided a user's search query and a set of search result documents.
        Your task is to provide a concise, summarized answer to the user's query with the help of the provided the search results.

        The user's query will be included in the <user_query> tag.
        The search results will be provided in JSON format in the <search_results> tag.
        
        Here are the critical rules that you MUST abide by:
        - You must only use the provided search result documents to generate your summary. Do not fabricate any answers or use your prior knowledge.
        - You must only summarize the search result documents that are relevant to the user's query. Do not reference any search results that aren't related to the user's query. If none of the provided search results are relevant to the user's query, reply with  "My apologies, I am unable to answer that question with the provided search results".
        - You must keep your summary to less than 10 sentences. You are encouraged to use bulleted lists, where stylistically appropriate.
        - Only respond with the summary without any extra explantion. Do not include any sentences like 'Sure, here is an explanation...'.

        <user_query>
        {query}
        </user_query>

        <search_results>
        {search_result_str}
        </search_results>

        [/INST]
    """
    
    resp = complete(st.session_state.model, prompt)
    return resp

def display_summary(summary):
    """
    Display the AI summary in the UI
    """
    st.subheader("AI summary")
    container = st.container(border=True)
    container.markdown(summary)

def display_search_results(results, search_col):
    """
    Display the search results in the UI
    """
    st.subheader("Search results")
    for i, result in enumerate(results):
        container = st.expander(f"Result {i+1}", expanded=True)
        container.markdown(result[search_col])

def main():
    init_layout()
    init_config_options()
    
    # run chat engine
    if not st.session_state.query:
        return
    results = query_cortex_search_service(st.session_state.query)
    search_col = get_search_column(st.session_state.cortex_search_service)
    if st.session_state.summarize:
        with st.spinner("Summarizing results..."):
            summary = summarize_search_results(results, st.session_state.query, search_col)
            display_summary(summary)
    display_search_results(results, search_col)


if __name__ == "__main__":
    st.set_page_config(page_title="Cortex AI Search and Summary", layout="wide")
    
    session = make_session()
    root = Root(session)
    db, schema = session.get_current_database(), session.get_current_schema()
    
    main()