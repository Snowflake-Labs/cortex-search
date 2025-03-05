# Standard library imports
import functools
import hashlib
import json
import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import os

# Third-party library imports
import pandas as pd
import skopt
import streamlit as st
from skopt.space import Real

# Snowflake-specific imports
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSessionException
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.functions import (
    col,
    max as spmax,
    udf,
    lit,
)
from snowflake.snowpark.session import Session
from snowflake.snowpark.dataframe import DataFrame, Column
from snowflake.snowpark.types import StringType
from snowflake.snowpark.row import Row
from snowflake.core import Root

# --- Constants ---
RUN_ID = "RUN_ID"
DOC_ID = "DOC_ID"
RANK = "RANK"
DEBUG_SIGNALS = "DEBUG_SIGNALS"
RESPONSE_RESULTS = "RESPONSE_RESULTS"
QUERY_ID = "QUERY_ID"
QUERY = "QUERY"
CSS = "CORTEX SEARCH SERVICE"
DEBUG_PER_RESULT = "@DEBUG_PER_RESULT"
HIT_RATE = "HIT_RATE"
SDCG = "SDCG"
PRECISION = "PRECISION"
VALID_METRICS = (HIT_RATE, SDCG, PRECISION)
VALID_LIMITS = (1, 3, 5, 10, 20, 50, 100)
IDCG = {k: sum(1.0 / math.log2(i + 2.0) for i in range(k)) for k in VALID_LIMITS}
RELEVANCY = "RELEVANCY"
FINAL_SCORE = "FINAL_SCORE"
RELEVANCY_SCORE = "RELEVANCY_SCORE"
TOTAL_QUERIES = "TOTAL_QUERIES"
COMMENT = "COMMENT"
RUN_METADATA = "RUN_METADATA"
JUDGE_OUTPUT = "JUDGE_OUTPUT"
PROMPT = "PROMPT"
COMPLETE_FUNC = "SNOWFLAKE.CORTEX.COMPLETE"
LLM_MODEL = "llama3.1-405b"
TABLE_INPUT_PLACEHOLDER = "in format db.schema.table"
QUERIES_TABLE_COLUMNS = (QUERY,)  # ',' allows casting this into a set later
RELEVANCY_TABLE_COLUMNS = (
    QUERY,
    RELEVANCY,
)  # Third column of relevancy table can be either of DOC_ID or TEXT
SCRAPE_TABLE_COLUMNS = (
    RUN_ID,
    QUERY_ID,
    DOC_ID,
    RANK,
    DEBUG_SIGNALS,
)
METRICS_TABLE_COLUMNS = (
    RUN_ID,
    RUN_METADATA,
    HIT_RATE,
    SDCG,
    PRECISION,
)
lightgreen = "lightgreen"
lightyellow = "lightyellow"
lightcoral = "lightcoral"
AUTOTUNE_SCRAPE_LIMIT = 10
AUTOTUNE_SCRAPE_RELEVANCY_SCRAPE_LIMIT = 40
AUTOTUNE_SCRAPE_RELEVANCY_RERANK_DEPTH = 64
DEBUG = "debug"
SLOWMODE = "slowmode"
RERANK_WEIGHTS = "RerankWeights"
RerankDepth = "RerankingDepth"
RerankingMultiplier = "RerankingMultiplier"
EmbeddingMultiplier = "EmbeddingMultiplier"
TopicalityMultiplier = "TopicalityMultiplier"
GP_SEARCH_SPACE = [
    Real(0.02, 5.0, base=2.0, prior="log-uniform", name="reranking_multiplier"),
    Real(0.02, 5.0, base=2.0, prior="log-uniform", name="topicality_multiplier"),
]
INITIAL_GP_POINT = [
    [1.4, 1.0]
]  # reranking_multiplier, reranking_multiplier, topicality_multiplier
NUM_OF_INITIAL_EXPLORATION = 3
DEFAULT_EMBEDDING_MULTIPLIER = 1.0


# --- Utility Functions ---


def get_llm_judge_prompt():
    gaurav_prompt = """You are an expert search result rater. You are given a user query and a search result. Your task is to rate the search result based on its relevance to the user query. You should rate the search result on a scale of 0 to 3, where:
    0: The search result has no relevance to the user query.
    1: The search result has low relevance to the user query. In this case the search result may contain some information which seems very slightly related to the user query but not enough information to answer the user query. The search result contains some references or very limited information about some entities present in the user query. In case the query is a statement on a topic, the search result should be tangentially related to it.
    2: The search result has medium relevance to the user query. If the user query is a question, the search result may contain some information that is relevant to the user query but not enough information to answer the user query. If the user query is a search phrase/sentence, either the search result is centered around about most but not all entities present in the user query, or if all the entities are present in the result, the search result while not being centered around it has medium level of relevance. In case the query is a statement on a topic, the search result should be related to the topic.
    3: The search result has high relevance to the user query. If the user query is a question, the search result contains information that can answer the user query. Otherwise if the search query is a search phrase/sentence, it provides relevant information about all entities that are present in the user query and the search result is centered around the entities mentioned in the query. In case the query is a statement on a topic, the search result should be either be directly addressing it or be on the same topic.
    
    You should think step by step about the user query and the search result and rate the search result. You should also provide a reasoning for your rating.
    
    Use the following format:
    Rating: Example Rating
    Reasoning: Example Reasoning
    
    ### Examples
    Example:
    Example 1:
    INPUT:
    User Query: What is the definition of an accordion?
    Search Result: Accordion definition, Also called piano accordion. a portable wind instrument having a large bellows for forcing air through small metal reeds, a keyboard for the right hand, and buttons for sounding single bass notes or chords for the left hand. a similar instrument having single-note buttons instead of a keyboard.
    OUTPUT:
    Rating: 3
    Reasoning: In this case the search query is a question. The search result directly answers the user question for the definition of an accordion, hence it has high relevance to the user query.
    
    Example 2:
    INPUT:
    User Query: dark horse
    Search Result: Darkhorse is a person who everyone expects to be last in a race. Think of it this way. The person who looks like he can never get laid defies the odds and gets any girl he can by being sly,shy and cunning. Although he\'s not a player, he can really charm the ladies.
    OUTPUT:
    Rating: 3
    Reasoning: In this case the search query is a search phrase mentioning \'dark horse\'. The search result contains information about the term \'dark horse\' and provides a definition for it and is centered around it. Hence it has high relevance to the user query.
    
    Example 3:
    INPUT:
    User Query: Global warming and polar bears
    Search Result: Polar bear The polar bear is a carnivorous bear whose native range lies largely within the Arctic Circle, encompassing the Arctic Ocean, its surrounding seas and surrounding land masses. It is a large bear, approximately the same size as the omnivorous Kodiak bear (Ursus arctos middendorffi).
    OUTPUT:
    Rating: 2
    Reasoning: In this case the search query is a search phrase mentioning two entities \'Global warming\' and \'polar bears\'. The search result contains is centered around the polar bear which is one of the two entities in the search query. Therefore it addresses most of the entities present and hence has medium relevance. 
    
    Example 4:
    INPUT:
    User Query: Snowflake synapse private link
    Search Result: "This site can\'t be reached" error when connecting to Snowflake via Private Connectivity\nThis KB article addresses an issue that prevents connections to Snowflake failing with: "This site can\'t be reached" ISSUE: Attempting to reach Snowflake via Private Connectivity fails with the "This site can\'t be reached" error
    OUTPUT:
    Rating: 1
    Reasoning: In this case the search result is a search query mentioning \'Snowflake synapse private link\'. However the search result doesn\'t contain information about it. However it shows an error message for a generic private link which is tangentially related to the query, since snowflake synapse private link is a type of private link. Hence it has low relevance to the user query.
    
    Example 5:
    INPUT:
    User Query: The Punisher is American.
    Search Result: The Rev(Samuel Smith) is a fictional character, a supervillain appearing in American comic books published by Marvel Comics. Created by Mike Baron and Klaus Janson, the character made his first appearance in The Punisher Vol. 2, #4 (November 1987). He is an enemy of the Punisher.
    OUTPUT:
    Rating: 1
    Reasoning: In this case the search query is a statement concerning the Punisher. However the search result is about a character called Rev, who is an enemy of the Punisher. The search result is tangentially related to the user query but does not address topic about Punisher being an American. Hence it has low relevance to the user query.

    Example 6:
    INPUT:
    User Query: query_history
    Search Result: The function task_history() is not enough for the purposes when the required result set is more than 10k.If we perform UNION between information_schema and account_usage , then we will get more than 10k records along with recent records as from information_schema.query_history to snowflake.account_usage.query_history is 45 mins behind.
    OUTPUT:
    Rating: 1
    Reasoning: In this case the search query mentioning one entity \'query_history\'. The search result is neither centered around it and neither has medium relevance, it only contains an unimportant reference to it. Hence it has low relevance to the user query.
    
    Example 7:
    INPUT:
    User Query: Who directed pulp fiction?
    Search Result: Life on Earth first appeared as early as 4.28 billion years ago, soon after ocean formation 4.41 billion years ago, and not long after the formation of the Earth 4.54 billion years ago.
    OUTPUT:
    Rating: 0
    Reasoning: In the case the search query is a question. However the search result does is completely unrelated to it. Hence the search result is completely irrelevant to the movie pulp fiction. 
    ###
    
    Now given the user query and search result below, rate the search result based on its relevance to the user query and provide a reasoning for your rating.
    INPUT:
    User Query: {query}
    Search Result: {passage}
    OUTPUT:\n
"""

    return gaurav_prompt


def hit_rate(results: List[str], golden_to_score: Dict[str, Dict[str, int]]) -> int:
    """Calculate hit rate for search results."""
    for result in results:
        if result in golden_to_score and golden_to_score[result]["score"] > 0:
            return 1
    return 0


def sdcg(
    idcg_factor: float, results: List[str], golden_to_score: Dict[str, Dict[str, int]]
) -> float:
    """Calculate sDCG, a modified verion of nDCG for LLM Judge evaluation results."""
    k = len(results)

    if k not in IDCG:
        IDCG[k] = sum(1.0 / math.log2(i + 2.0) for i in range(k))
    ideal = idcg_factor * IDCG[k]

    if ideal == 0:
        return 0.0
    return _dcg(results, golden_to_score) / ideal


def _dcg(results: List[str], golden_to_score: Dict[str, Dict[str, int]]) -> float:
    """Calculate Discounted Cumulative Gain (DCG)."""
    return sum(
        golden_to_score[result]["score"] / math.log2(i + 2.0)
        for i, result in enumerate(results)
        if result in golden_to_score
    )


def count_relevant_results(
    results: List[str], golden_to_score: Dict[str, Dict[str, int]]
) -> int:
    """Helper function to count the number of relevant documents returned."""
    return sum(
        1
        for result in results
        if result in golden_to_score and golden_to_score[result]["score"] > 0
    )


def precision(results: List[str], golden_to_score: Dict[str, Dict[str, int]]) -> float:
    """Calculate precision for search results."""
    return (
        count_relevant_results(results, golden_to_score) / len(results)
        if results
        else 0.0
    )


def compute_display_metrics(
    retrieval_metrics: List[Dict[str, Dict[str, float]]], result_limit: int
) -> Dict[str, float]:
    """
    Compute display metrics by averaging valid metrics that fall within the result limit.

    Args:
        retrieval_metrics (list): List of metric dictionaries for each query.
        result_limit (int): The upper limit of results to consider for metrics.

    Returns:
        dict: Averaged metrics for display.
    """
    if not retrieval_metrics:
        return {}

    return {
        f"{metric_name}@{limit}": round(
            sum(rm[metric_name][str(limit)] for rm in retrieval_metrics)
            / len(retrieval_metrics),
            2,
        )
        for metric_name in VALID_METRICS
        for limit in VALID_LIMITS
        if limit <= result_limit
    }


def calculate_metrics(
    scraped_results: List[str],
    golden_to_score: Dict[str, Dict[str, int]],
    metrics: Dict[str, Callable[[List[str], Dict[str, Dict[str, int]]], Any]],
    cutoff_values: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics based on scraped results and their comparison with golden scores.

    Args:
        scraped_results (list): List of document IDs returned by the search.
        golden_to_score (dict): Mapping of document IDs to their relevance scores.
        metrics (dict): Metric functions (precision, hitrate, etc.) to apply.
        cutoff_values (list): Cutoff values (e.g., top-k limits) to compute metrics at.

    Returns:
        dict: Metrics for each cutoff value.
    """
    return {
        metric_name: {
            str(cutoff): metrics[metric_name](scraped_results[:cutoff], golden_to_score)
            for cutoff in cutoff_values
        }
        for metric_name in metrics
    }


def calculate_average_metrics(
    retrieval_metrics: List[Dict[str, Dict[str, float]]], metric_name: str
) -> Dict[str, float]:
    """
    Calculate average metrics for a given metric name across multiple queries.

    Args:
        retrieval_metrics (list): List of per-query retrieval metrics.
        metric_name (str): The name of the metric to average (e.g., precision).

    Returns:
        dict: Averaged metric values across different cutoff points.
    """
    if not retrieval_metrics:
        return {str(cutoff): 0 for cutoff in VALID_LIMITS}

    return {
        str(cutoff): sum(rm[metric_name][str(cutoff)] for rm in retrieval_metrics)
        / len(retrieval_metrics)
        for cutoff in VALID_LIMITS
    }


def extract_and_sort_metrics(df: pd.DataFrame) -> pd.Series:
    """
    Extract and sort metrics from a Snowflake DataFrame for further processing.

    Args:
        df (pd.DataFrame): Snowflake DataFrame containing retrieval metrics.

    Returns:
        pd.Series: A series with selected and sorted metrics.
    """
    selected_df = pd.DataFrame(df[QUERY_ID])
    selected_df[QUERY] = df[QUERY]

    k_value = str(st.session_state.k_value)
    for metric in VALID_METRICS:
        df_expanded = df[metric].apply(lambda x: json.loads(x)).apply(pd.Series)
        if k_value in df_expanded.columns:
            df_expanded = df_expanded[[k_value]]
            df_expanded.columns = [f"{metric}@{k_value}"]

        selected_df = pd.concat([selected_df, df_expanded], axis=1)

    return selected_df


def color_code_columns(row: pd.Series, columns: List[str]) -> List[str]:
    """
    Apply color coding to specific columns in a row based on query and document IDs.

    Args:
        row (pd.Series): A row of data (typically from a DataFrame).
        columns (list): List of column names to apply color coding to.

    Returns:
        list: List of styles for each column in the row.
    """
    styles = [""] * len(row)

    if str(row[QUERY_ID]) in st.session_state.colors:
        doc_colors = st.session_state.colors[str(row[QUERY_ID])]
        doc_id = str(row[DOC_ID])
        if doc_id in doc_colors:
            color = doc_colors[doc_id]
            for column in columns:
                column_index = row.index.get_loc(column)
                styles[column_index] = f"background-color: {color}"

    return styles


def get_column_names(fqn: str, session: Session) -> Set[str]:
    """
    Verify that the specified table contains all required columns.

    Parameters:
    - fqn: str - Fully qualified name of the table (e.g., "database.schema.table")
    - required_cols: Set[str] - A set of column names that must be present in the table

    Returns:
    - bool: True if all required columns are present in the table, False otherwise
    """
    try:
        result = session.sql(f"SHOW COLUMNS IN TABLE {fqn}").collect()
        actual_columns = {row[2] for row in result}
        return actual_columns
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return {}


def get_search_column() -> str:
    """
    Verify that the specified table contains all required columns.

    Parameters:
    - fqn: str - Fully qualified name of the table (e.g., "database.schema.table")
    - required_cols: Set[str] - A set of column names that must be present in the table

    Returns:
    - bool: True if all required columns are present in the table, False otherwise
    """
    try:
        result = session.sql(
            f"DESCRIBE CORTEX SEARCH SERVICE {st.session_state.css_fqn}"
        ).collect()
        return result[0][5]
    except Exception as e:
        st.write(f"An error occurred: {e}")
        st.stop()
        return ""


def check_table_exists(fqn: str) -> bool:
    """
    Check whether a table exists in the specified database and schema.

    Args:
        database (str): The name of the database.
        schema (str): The name of the schema.
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    try:
        session.sql(
            f"""
            DESCRIBE TABLE {fqn}
            """
        ).collect()
        return True
    except Exception as _:
        return False


def check_cortex_service_exists(fqn: str) -> bool:
    """
    Check whether a fully qualified Cortex Search service exists in Snowflake.

    Args:
        fqn (str): Fully qualified name of the Cortex Search service.

    Returns:
        bool: True if the service exists, False otherwise.
    """
    query = f"DESCRIBE CORTEX SEARCH SERVICE {fqn}"
    try:
        result = session.sql(query).collect()
        return len(result) > 0
    except SnowparkSQLException as e:
        st.error(f"Error checking Cortex Search service existence {fqn}: {e}")
        return False


def validate_fqn(fqn: str, fqn_name: str) -> None:
    """
    Validate the fully qualified name (FQN) structure.

    Parameters:
    - fqn (str): Fully Qualified Name in the format `DB.SCHEMA.OBJECT`.
    - fqn_name (str): The name of the input field being validated.

    Raises:
    - AssertionError: If validation fails for FQN structure.
    """

    # Check if FQN is provided
    assert fqn, f"Please provide the {fqn_name} fully qualified name before proceeding."

    # Split FQN and validate structure
    parts = fqn.upper().split(".")
    assert (
        len(parts) == 3
    ), f"{fqn_name} must be a fully qualified name in the format `DB.SCHEMA.OBJECT`."

    # Validate each part of the FQN
    for part in parts:
        assert part, f"Each part of the {fqn_name} fully qualified name must be non-empty (e.g., `DB`, `SCHEMA`, `OBJECT`)."


def validate_doc_id_col(fqn):
    assert check_table_exists(fqn), f"The table `{fqn}` does not exist!"
    actual_cols = get_column_names(fqn, session)
    if DOC_ID in actual_cols:
        st.warning(
            "Warning: user-defined `DOC_ID` should not exist in relevancy table. Remove the `DOC_ID` column if it is NOT from the relevancy table generated by this tool"
        )


def validate_table(
    fqn: str,
    fqn_name: str,
    session: Session,
    must_exist: bool = True,
    required_cols: Set[str] = set(),
) -> None:
    """
    Validate the fully qualified name (FQN) structure and existence for a table.

    Parameters:
    - fqn (str): Fully Qualified Name in the format `DB.SCHEMA.TABLE`.
    - fqn_name (str): The name of the input field being validated.
    - required_cols (Set[str]): Columns that must be present in the table, if applicable.
    - must_exist (bool): If True, checks for the existence of the FQN.

    Raises:
    - AssertionError: If validation fails for FQN structure, existence, or required columns.
    """

    # Validate the FQN structure
    validate_fqn(fqn, fqn_name)

    if must_exist:
        assert check_table_exists(fqn), f"The table `{fqn}` does not exist!"
        actual_cols = get_column_names(fqn, session)
        missing_cols = required_cols - actual_cols
        assert not missing_cols, f"Columns missing from `{fqn}` table: {missing_cols}"

    st.write(f"{fqn_name} = `{fqn}` ✅")


def validate_css(
    fqn: str,
) -> None:
    """
    Validate the fully qualified name (FQN) structure and existence for a Cortex Search Service.

    Parameters:
    - fqn (str): Fully Qualified Name in the format `DB.SCHEMA.SERVICE`.

    Raises:
    - AssertionError: If validation fails for FQN structure or existence.
    """

    # Validate the FQN structure
    validate_fqn(fqn, CSS)
    assert check_cortex_service_exists(fqn), f"The {CSS} `{fqn}` does not exist!"
    st.write(f"{CSS} = `{fqn}` ✅")


def md5_hash_string(session: Session) -> udf:
    """Create a UDF for generating an MD5 hash from a string."""

    def _md5_hash_string(s: str) -> str:
        # Validate input type
        if not isinstance(s, str):
            raise TypeError(f"Expected input of type str, but got {type(s).__name__}")
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    # Define and return the UDF with Snowflake's session
    return udf(_md5_hash_string, return_type=StringType(), session=session)


def generate_runid() -> str:
    """Generate a unique run ID based on the current timestamp."""
    return hashlib.md5(str(datetime.now()).encode("utf-8")).hexdigest()


def prepare_query_df(session: Session) -> DataFrame:
    """Prepare the query table and ensure QUERY_ID column exists."""
    query_df = session.table(st.session_state.queryset_fqn)

    # Convert to DataFrame
    if QUERY_ID not in query_df.columns:
        query_df = query_df.with_column(
            QUERY_ID, st.session_state.md5_hash(query_df[QUERY])
        )  # Generate QUERY_ID if missing

    return query_df


def perform_scrape(
    query_df: DataFrame,
    root: Root,
    autotune: bool = False,
    experimental_params: dict = {},
    run_id: str = "",
) -> Dict[str, Any]:
    """Perform scraping for each query and return the results."""
    css_db, css_schema, css_service = st.session_state.css_fqn.split(".")
    svc = root.databases[css_db].schemas[css_schema].cortex_search_services[css_service]
    scrape_out = dict()
    all_queries = query_df.collect()
    validate_queryset(all_queries)

    total_rows = len(all_queries)
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Scraping in progress.. (this may take a while)")

    result_limit = AUTOTUNE_SCRAPE_LIMIT if autotune else st.session_state.result_limit
    if st.session_state.scrape_for_autotune_relevancy:
        result_limit = AUTOTUNE_SCRAPE_RELEVANCY_SCRAPE_LIMIT
        experimental_params = {
            DEBUG: True,
            SLOWMODE: True,
            RERANK_WEIGHTS: {RerankDepth: AUTOTUNE_SCRAPE_RELEVANCY_RERANK_DEPTH},
        }
        st.write("Run scape for autotune's LLM generated relevancy with")
        st.write("1. result limit: " + str(result_limit))
        st.write("2. experimental params: " + str(experimental_params))

    for i, row in enumerate(all_queries):
        progress_percentage = int((i + 1) / total_rows * 100)
        progress_bar.progress(progress_percentage)
        query, query_id = row[QUERY], str(row[QUERY_ID])

        # Perform the search request
        columns = [st.session_state.css_text_col]
        response = svc.search(
            query,
            limit=result_limit,
            columns=columns + st.session_state.additional_columns,
            filter=st.session_state.filter,
            # debug set to True to display debug signals
            experimental=experimental_params
            if experimental_params
            else {DEBUG: True, SLOWMODE: True},
        )
        scrape_out[query_id] = {
            QUERY: query,
            RUN_ID: (run_id or st.session_state.scrape_run_id),
            RESPONSE_RESULTS: response.results,
        }
    status_text.text("Scrape Finished!")
    return scrape_out


def generate_docid(doc_text: str) -> str:
    "Return the doc-id for the doc in service's response result."
    return hashlib.md5(doc_text.encode("utf-8")).hexdigest()


def perform_scrape_for_autotune(
    query_df: DataFrame,
    root: Root,
    experimental_params: dict = {},
    run_id: str = "",
) -> Dict[str, Any]:
    """Perform scraping for each query in autotune process."""

    raw_scrape_out = perform_scrape(
        query_df,
        root,
        autotune=True,
        experimental_params=experimental_params,
        run_id=run_id,
    )
    return {
        query_id: [
            generate_docid(str(result[st.session_state.css_text_col]))
            for result in response[RESPONSE_RESULTS]
        ]
        for query_id, response in raw_scrape_out.items()
    }


def perform_scrape_for_eval(
    session: Session,
    query_df: DataFrame,
    root: Root,
    experimental_params: dict = {},
    run_id: str = "",
) -> DataFrame:
    """Perform scraping for each query in evaluation process."""

    raw_scrape_out = perform_scrape(
        query_df,
        root,
        autotune=False,
        experimental_params=experimental_params,
        run_id=run_id,
    )

    scrape_out = []
    for query_id, response in raw_scrape_out.items():
        # Append each result to the scrape output
        for rank, result in enumerate(response[RESPONSE_RESULTS]):
            scrape_out.append(
                {
                    QUERY: response[QUERY],
                    RUN_ID: response[RUN_ID],
                    QUERY_ID: query_id,
                    DOC_ID: generate_docid(str(result[st.session_state.css_text_col])),
                    RANK: rank + 1,
                    DEBUG_SIGNALS: result[DEBUG_PER_RESULT],
                    st.session_state.css_text_col.upper(): result[
                        st.session_state.css_text_col
                    ],  # Todo(amehta): Handle markdown if necessary
                }
            )
    scrape_df = session.create_dataframe(scrape_out)
    return scrape_df


def store_scrape_results(scrape_df: DataFrame) -> None:
    """Store scrape results in the Snowflake table and display progress."""
    scrape_df.write.mode("append").save_as_table(st.session_state.scrape_fqn)
    # st.success(f"""
    #            Finished scrape in {round(duration.total_seconds(),1)} seconds. Generated Scrape stored in {st.session_state.scrape_fqn} with Run ID: {st.session_state.scrape_run_id}
    #            """)


def generate_and_store_scrape(session: Session, root: Root) -> None:
    """Generate and store scrape results."""
    st.session_state.start_time = datetime.now()

    query_df = prepare_query_df(session)
    st.session_state.scrape_for_autotune_relevancy = st.session_state.run_autotuning
    scrape_df = perform_scrape_for_eval(session, query_df, root)
    st.session_state.scrape_for_autotune_relevancy = False

    store_scrape_results(scrape_df)


def refresh_state_to_pre_eval():
    st.session_state.rel_scores = {}
    st.session_state.colors = {}


def run_eval(relevancy_fqn: str, result_fqn: str, run_comment: str) -> None:
    """Run evaluation on scraped results against golden standards."""
    start_time = datetime.now()

    # Reset progress and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        # Clear session state for a fresh run
        refresh_state_to_pre_eval()
        status_text.text("Initializing tables...")
        query_df = prepare_query_df(session)
        scrape_df = prepare_scrape_df(session)

        # scrape_df.collect()
        progress_bar.progress(10)  # 10% done

        # Validate scrape
        status_text.text("Validating scraped data...")
        validate_scrape(scrape_df)
        progress_bar.progress(20)  # 20% done

        # Generate relevancy table
        status_text.text("Preparing relevancy table...")
        if st.session_state.relevancy_provided:
            relevancy_df = prepare_relevancy_df(relevancy_fqn, session)
        else:
            relevancy_df = prepare_relevancy_df_llm(scrape_df)

        progress_bar.progress(40)  # 40% done
        # Prepare golden scores
        status_text.text("Extracting and preparing golden scores...")
        raw_goldens = extract_and_dedupe_goldens(relevancy_df)
        goldens = prepare_golden_scores(raw_goldens)
        progress_bar.progress(60)  # 60% done

        # Evaluate queries
        status_text.text("Evaluating queries...")
        retrieval_metrics = evaluate_queries(query_df, scrape_df, goldens)
        progress_bar.progress(90)  # 90% done

        # Persist results
        status_text.text("Saving evaluation metrics...")
        persist_metrics(
            retrieval_metrics, relevancy_fqn, result_fqn, run_comment, scrape_df
        )
        progress_bar.progress(100)  # 100% done

        # Finalize
        duration = datetime.now() - start_time
        st.session_state.scrape_df = scrape_df
        st.session_state.queryid_to_query = {
            str(row[QUERY_ID]): row[QUERY] for row in query_df.collect()
        }
        status_text.text("Evaluation Complete!")

        eval_output = f"""
            **Evaluation finished in {round(duration.total_seconds(), 1)} seconds**
            - **Aggregate Eval metrics** are added to `{result_fqn}`
            - **Per Query Metrics** are added to `{result_fqn}_PERQUERY`
            """
        # todo (amehta): add llm relevancy table one-liner summary
        st.success(eval_output)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e


def compute_fusion_score_from_service(
    root: Any,
    query_df: DataFrame,
    goldens: Dict[str, Dict[str, Dict[str, int]]],
    write_result: bool,
    experimental_params: Dict[str, Any],
):
    """Scrape search service with config from params for avg sdcg@10 of goldens."""

    scrape_out = perform_scrape_for_autotune(
        query_df, root, experimental_params=experimental_params
    )

    avg_sdcg_10 = 0
    total_queries = len(query_df.collect())
    for row in query_df.collect():
        query_id = row[QUERY_ID]
        scraped_results = list(dict.fromkeys(scrape_out[query_id]))
        avg_sdcg_10 += sdcg(
            idcg_factor=st.session_state.idcg_factor,
            results=scraped_results[:10],  # GRID_SEARCH_METRICS = "SDCG@10"
            golden_to_score=goldens[query_id],
        )
    avg_sdcg_10 /= total_queries
    if write_result:
        st.write("Result for config on test set:")
        st.text(json.dumps(experimental_params, indent=4))
        st.success(f"Avg SDCG@10 on Test Set: {round(avg_sdcg_10, 4)}")
        st.divider()
    return avg_sdcg_10


def run_autotuning(
    relevancy_fqn: str,
    autotuning_result_fqn: str,
    run_comment: str,
    num_calls_to_gp_minimize: int = 21,
) -> None:
    """Run autotuning on scraped results against golden standards."""
    start_time = datetime.now()
    status_text = st.empty()

    def _run_autotuning():
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state.result_limit = AUTOTUNE_SCRAPE_LIMIT

        status_text.text("Initializing tables...")
        query_df = prepare_query_df(session)
        optimized_query_df, test_query_df = query_df.random_split([0.8, 0.2])
        # Generate relevancy table
        progress_bar.progress(10)  # 10% done
        status_text.text("Preparing relevancy table...")
        if st.session_state.relevancy_provided:
            relevancy_df = prepare_relevancy_df(relevancy_fqn, session)
        else:
            scrape_df = prepare_scrape_df(session)
            # validate_scrape(scrape_df)
            relevancy_df = prepare_relevancy_df_llm(scrape_df)

        progress_bar.progress(40)  # 20% done
        raw_goldens = extract_and_dedupe_goldens(relevancy_df)
        goldens = prepare_golden_scores(raw_goldens)
        progress_bar.progress(60)  # 60% done
        compute_optimized_fusion_score_from_service_partial = functools.partial(
            compute_fusion_score_from_service, root, optimized_query_df, goldens, False
        )
        compute_test_fusion_score_from_service_partial = functools.partial(
            compute_fusion_score_from_service, root, test_query_df, goldens, True
        )

        params_to_test_sdcg = dict()

        @skopt.utils.use_named_args(GP_SEARCH_SPACE)
        def compute_fusion_score(
            reranking_multiplier: float,
            topicality_multiplier: float,
        ):
            params = {
                RERANK_WEIGHTS: {
                    RerankingMultiplier: reranking_multiplier,
                    EmbeddingMultiplier: DEFAULT_EMBEDDING_MULTIPLIER,
                    TopicalityMultiplier: topicality_multiplier,
                },
                DEBUG: True,
                SLOWMODE: True,
            }

            status_text.text("Scraping on Train Set:")
            optimized_sdcg = compute_optimized_fusion_score_from_service_partial(params)
            status_text.text("Scraping on Test Set:")
            test_sdcg = compute_test_fusion_score_from_service_partial(params)
            params_to_test_sdcg[json.dumps(params, indent=4)] = test_sdcg

            # Need this formula because using gp_minimize (not maximization).
            return 1.0 - optimized_sdcg

        status_text.text("Finding the best parameters.. ")
        _ = skopt.gp_minimize(
            compute_fusion_score,
            GP_SEARCH_SPACE,
            n_calls=num_calls_to_gp_minimize,
            n_initial_points=NUM_OF_INITIAL_EXPLORATION,
            x0=INITIAL_GP_POINT,
        )
        progress_bar.progress(90)  # 90% done
        autotune_params_json = max(
            params_to_test_sdcg.keys(), key=lambda k: params_to_test_sdcg[k]
        )
        st.write("Best Performing Config:")
        st.success(autotune_params_json)
        st.success(
            f"Avg SDCG@10 on Test Set: {round(params_to_test_sdcg[autotune_params_json], 4)}"
        )
        run_id = generate_runid()
        optimal_scrape_df = perform_scrape_for_eval(
            session,
            query_df,
            root,
            experimental_params=json.loads(autotune_params_json),
            run_id=run_id,
        )

        # Evaluate queries
        status_text.text("Evaluating queries...")
        retrieval_metrics = evaluate_queries(query_df, optimal_scrape_df, goldens)

        # Persist results
        status_text.text("Saving evaluation metrics...")
        persist_metrics(
            retrieval_metrics,
            relevancy_fqn,
            autotuning_result_fqn,
            run_comment,
            optimal_scrape_df,
            autotune=True,
            autotuned_params=json.loads(autotune_params_json),
        )
        progress_bar.progress(100)  # 100% done

    # Run the autotuning process and return any error if occurs.
    try:
        _run_autotuning()

        # Finalize
        duration = datetime.now() - start_time
        status_text.text("Autotuning Complete!")
        st.success(
            f"""
            **Autotuning finished in {round(duration.total_seconds(), 1)} seconds**

            - **Aggregate Eval metrics** are added to `{autotuning_result_fqn}`
            - **Per Query Metrics** are added to `{autotuning_result_fqn}_PERQUERY`
            """,
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e


def prepare_scrape_df(session: Session) -> DataFrame:
    """Prepare the scrape df to choose rows for the current RUN ID."""
    scrape_table = session.table(st.session_state.scrape_fqn)
    scrape_df = scrape_table.filter(col(RUN_ID) == st.session_state.scrape_run_id)

    return scrape_df


def validate_scrape(scrape_df: DataFrame) -> None:
    """Validate non-empty scrape DataFrame."""
    assert (
        scrape_df.count() != 0
    ), "Scrape is empty! Recheck the Run ID or the Scrape table."


def validate_queryset(queries: List[Row]) -> None:
    """Validate non-empty queries."""
    assert len(queries) != 0, "Queries Table is empty! Recheck the Query table"


def get_result_limit(scrape_df: DataFrame) -> int:
    """Get the result limit based on scrape results."""
    return scrape_df.select(spmax("rank")).collect()[0][0]


def prepare_relevancy_df(relevancy_fqn: str, session: Session) -> DataFrame:
    """Prepare the relevancy table, ensuring QUERY_ID is present."""
    relevancy_df = session.table(relevancy_fqn)

    if QUERY_ID in relevancy_df.columns:
        # Ensure QUERY_ID is of string type
        relevancy_df = relevancy_df.withColumn(
            QUERY_ID, relevancy_df[QUERY_ID].cast("string")
        )
    else:
        # Add QUERY_ID if not present
        relevancy_df = relevancy_df.withColumn(
            QUERY_ID, st.session_state.md5_hash(relevancy_df[QUERY])
        )

    if DOC_ID in relevancy_df.columns:
        # Ensure DOC_ID is of string type
        relevancy_df = relevancy_df.withColumn(
            DOC_ID, relevancy_df[DOC_ID].cast("string")
        )
    else:
        # Add DOC_ID if not present
        relevancy_df = relevancy_df.withColumn(
            DOC_ID,
            st.session_state.md5_hash(relevancy_df[st.session_state.css_text_col]),
        )

    return relevancy_df


def output_prompt_col(query_col: Column, passage_col: Column) -> Column:
    """
    Generate a Snowpark Column with the formatted LLM judge prompt.
    Dynamically replaces `{query}` and `{passage}` in the prompt with column values.
    """
    from snowflake.snowpark.functions import concat, lit

    # Split the prompt into parts before and after {query} and {passage}
    entire_prompt = get_llm_judge_prompt()
    prompt_before_query = entire_prompt.split("{query}")[0]
    prompt_after_query = entire_prompt.split("{query}")[1].split("{passage}")[0]
    prompt_after_passage = entire_prompt.split("{passage}")[1]

    # Concatenate parts of the prompt with column values
    return concat(
        lit(prompt_before_query),  # Text before {query}
        query_col,  # Dynamic query value
        lit(prompt_after_query),  # Text between {query} and {passage}
        passage_col,  # Dynamic passage value
        lit(prompt_after_passage),  # Text after {passage}
    )


def prepare_relevancy_df_llm(scrape_df: DataFrame) -> DataFrame:
    if check_table_exists(st.session_state.relevancy_fqn):
        st.success(
            f"Using existing relevancy table {st.session_state.relevancy_fqn} which was generated for scrape run ID: {st.session_state.scrape_run_id}"
        )
    else:
        temp_table_name = st.session_state.relevancy_fqn + "_temp"

        scrape_df_interm = scrape_df.withColumn(
            "prompt", output_prompt_col(col(QUERY), col(st.session_state.css_text_col))
        )

        scrape_df_interm.write.mode("overwrite").save_as_table(temp_table_name)

        session.sql(f"""
    CREATE OR REPLACE TABLE {st.session_state.relevancy_fqn} AS
    SELECT
        *,
        SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-405b',
            [{{'role': 'user', 'content': prompt}}],
            {{'temperature': 0, 'top_p': 1}}
        )['choices'][0]['messages']::VARCHAR AS LLM_JUDGE,
        REGEXP_SUBSTR(LLM_JUDGE, 'R[a-z]*g: ([0-9])', 1, 1, 'e', 1) AS RELEVANCY
    FROM {temp_table_name}
    WHERE REGEXP_SUBSTR(LLM_JUDGE, 'R[a-z]*g: ([0-9])', 1, 1, 'e', 1) IS NOT NULL
""").collect()
        st.success(
            f"LLM Judge generated relevancies are stored in: {st.session_state.relevancy_fqn}"
        )

    relevancy_df = session.table(st.session_state.relevancy_fqn)
    return relevancy_df


def extract_and_dedupe_goldens(
    relevancy_df: DataFrame,
) -> Dict[str, List[Tuple[str, int]]]:
    """Extract golden scores from the relevancy table."""
    raw_goldens: Dict[str, List[Tuple[str, int]]] = {}
    relevance_color_mapping = {
        0: lightcoral,
        1: lightyellow,
        2: lightgreen,
        3: lightgreen,
    }

    relevancies = relevancy_df.collect()
    for row in relevancies:
        rel_score = int(row[RELEVANCY])
        query_id, doc_id = str(row[QUERY_ID]), str(row[DOC_ID])

        # Dedup block:
        # Check if the current doc_id already exists for the query_id
        if (
            query_id in st.session_state.rel_scores
            and doc_id in st.session_state.rel_scores[query_id]
        ):
            # If the new relevance score is higher, update the score
            existing_score = st.session_state.rel_scores[query_id][doc_id]
            if rel_score > existing_score:
                st.session_state.colors[query_id][doc_id] = relevance_color_mapping.get(
                    rel_score, lightgreen
                )
                st.session_state.rel_scores[query_id][doc_id] = rel_score

                # Update the raw_goldens list with the new score for the doc_id
                for i, (d_id, _) in enumerate(raw_goldens[query_id]):
                    if d_id == doc_id:
                        raw_goldens[query_id][i] = (doc_id, rel_score)
                        break
        else:
            # If the doc_id is not a duplicate, store the new doc_id and its score
            st.session_state.colors.setdefault(query_id, {})[doc_id] = (
                relevance_color_mapping.get(rel_score, lightgreen)
            )
            st.session_state.rel_scores.setdefault(query_id, {})[doc_id] = rel_score
            raw_goldens.setdefault(query_id, []).append((doc_id, rel_score))

    return raw_goldens


def prepare_golden_scores(
    raw_goldens: Dict[str, List[Tuple[str, int]]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Prepare sorted golden scores for each query."""
    if st.session_state.relevancy_provided:
        st.session_state.idcg_factor = max(
            [
                max([score for _, score in raw_goldens[query_id]] or [0])
                for query_id in raw_goldens
            ]
            or [0]
        )
    return {
        query_id: {
            doc_id: {"rank": rank, "score": score}
            for rank, (doc_id, score) in enumerate(
                sorted(raw_goldens[query_id], key=lambda x: x[1], reverse=True)
            )
        }
        for query_id in raw_goldens
    }


def evaluate_queries(
    query_df: DataFrame,
    scrape_df: DataFrame,
    goldens: Dict[str, Dict[str, Dict[str, int]]],
) -> List[Dict[str, Any]]:
    """Evaluate all queries against the scraped results."""
    retrieval_metrics: List[Dict[str, Any]] = []

    queries = query_df.collect()

    for row in queries:
        query_id = str(row[QUERY_ID])

        # Preserve rank ordering as you filter
        scrape_for_query_df = scrape_df.filter(col(QUERY_ID) == query_id).order_by(
            col(RANK)
        )

        # Dedup doc_ids in result while persisting order
        scraped_results = list(
            dict.fromkeys(r[DOC_ID] for r in scrape_for_query_df.collect())
        )

        _sdcg = functools.partial(sdcg, st.session_state.idcg_factor)
        if query_id in goldens:
            golden_to_score = goldens[query_id]
            retrieval_metrics.append(
                {
                    QUERY_ID: query_id,
                    HIT_RATE: calculate_metrics(
                        scraped_results,
                        golden_to_score,
                        {HIT_RATE: hit_rate},
                        VALID_LIMITS,
                    )[HIT_RATE],
                    SDCG: calculate_metrics(
                        scraped_results,
                        golden_to_score,
                        {SDCG: _sdcg},
                        VALID_LIMITS,
                    )[SDCG],
                    PRECISION: calculate_metrics(
                        scraped_results,
                        golden_to_score,
                        {PRECISION: precision},
                        VALID_LIMITS,
                    )[PRECISION],
                }
            )

    return retrieval_metrics


def calculate_aggregate_metrics(
    retrieval_metrics: List[Dict[str, Any]],
    relevancy_fqn: str,
    run_comment: str,
    scrape_df: DataFrame,
    autotune: bool = False,
    autotuned_params: dict = {},
    run_id: str = "",
) -> Dict[str, Any]:
    """Calculate aggregate metrics from retrieval metrics."""
    if autotune:
        run_comment = run_comment or f"Autotuning with Params: {autotuned_params}"
    aggregate_metrics = {
        RUN_ID: (run_id or st.session_state.scrape_run_id),
        RUN_METADATA: {
            "Timestamp": datetime.now(),
            "QueryTable": st.session_state.queryset_fqn,
            "NumQueriesScraped": scrape_df.count(),
            "RelevancyTable": relevancy_fqn,
            "ScrapeTable": st.session_state.scrape_fqn,
            "CortexSearchService": st.session_state.css_fqn,
            "MetricsCalculatedAt": datetime.now(),
            "Comment": run_comment,
        },
        HIT_RATE: calculate_average_metrics(retrieval_metrics, HIT_RATE),
        SDCG: calculate_average_metrics(retrieval_metrics, SDCG),
        PRECISION: calculate_average_metrics(retrieval_metrics, PRECISION),
    }
    if autotuned_params:
        aggregate_metrics[RUN_METADATA]["AutotunedParams"] = autotuned_params
        aggregate_metrics[RUN_METADATA]["ScrapeTable"] = ""
    return aggregate_metrics


def create_display_metrics(
    aggregate_metrics: Dict[str, Any],
    retrieval_metrics: List[Dict[str, Any]],
    run_id: str = "",
) -> DataFrame:
    """Create display metrics DataFrame for user interface."""
    display_data = {
        RUN_ID: (run_id or st.session_state.scrape_run_id),
        RUN_METADATA: aggregate_metrics[RUN_METADATA],
        **compute_display_metrics(retrieval_metrics, st.session_state.result_limit),
    }
    return session.create_dataframe([display_data])


def save_per_query_metrics(
    retrieval_metrics: List[Dict[str, Any]], result_fqn: str
) -> None:
    """Save per-query metrics to the database."""
    per_query_df = session.create_dataframe(retrieval_metrics).withColumn(
        RUN_ID, lit(st.session_state.scrape_run_id)
    )
    st.session_state.retrieval_metrics_per_query_df = per_query_df
    per_query_table_name = f"{result_fqn}_PERQUERY"
    per_query_df.write.mode("append").save_as_table(per_query_table_name)


def calculate_and_save_aggregate_metrics(
    retrieval_metrics: List[Dict[str, Any]],
    relevancy_fqn: str,
    run_comment: str,
    scrape_df: DataFrame,
    result_fqn: str,
    autotune: bool,
    autotuned_params: Dict[str, Any],
    run_id: str,
) -> Dict[str, Any]:
    aggregate_metrics = calculate_aggregate_metrics(
        retrieval_metrics,
        relevancy_fqn,
        run_comment,
        scrape_df,
        autotune=autotune,
        autotuned_params=autotuned_params,
        run_id=run_id,
    )
    aggregate_metrics_df = session.create_dataframe([aggregate_metrics])
    aggregate_metrics_df.write.mode("append").save_as_table(result_fqn)
    return aggregate_metrics


def display_aggregate_metrics(
    aggregate_metrics: Dict[str, Any],
    retrieval_metrics: List[Dict[str, Any]],
    run_id: str,
    autotune: bool,
) -> None:
    display_metrics_df = create_display_metrics(
        aggregate_metrics, retrieval_metrics, run_id
    )
    if autotune:
        st.session_state.autotuning_aggregate_metrics_display_df = display_metrics_df
    else:
        st.session_state.aggregate_metrics_display_df = display_metrics_df


def persist_metrics(
    retrieval_metrics: List[Dict[str, Any]],
    relevancy_fqn: str,
    result_fqn: str,
    run_comment: str,
    scrape_df: DataFrame,
    autotune: bool = False,
    autotuned_params: dict[str, Any] = {},
    run_id: str = "",
) -> None:
    """Persist the metrics in the database."""

    if not autotune:
        save_per_query_metrics(retrieval_metrics, result_fqn)

    aggregate_metrics = calculate_and_save_aggregate_metrics(
        retrieval_metrics,
        relevancy_fqn,
        run_comment,
        scrape_df,
        result_fqn,
        autotune,
        autotuned_params,
        run_id,
    )
    display_aggregate_metrics(aggregate_metrics, retrieval_metrics, run_id, autotune)


def validate_scrape_fqn(must_exist: bool) -> None:
    if st.session_state.scrape_fqn != st.session_state.prev_scrape_fqn:
        required_scrape_cols = set(SCRAPE_TABLE_COLUMNS) | {
            st.session_state.css_text_col
        }
        validate_table(
            fqn=st.session_state.scrape_fqn,
            fqn_name="Scrape table",
            session=session,
            must_exist=must_exist,
            required_cols=required_scrape_cols,
        )
        st.session_state.prev_scrape_fqn = st.session_state.scrape_fqn
    else:
        st.write(f"Scrape table = `{st.session_state.scrape_fqn}` ✅")


def process_obtain_scrape() -> None:
    """Process and obtain scrape details from the user."""
    required_scrape_cols = set(SCRAPE_TABLE_COLUMNS) | {st.session_state.css_text_col}
    st.session_state.scrape_fqn = st.text_input(
        f"Enter Input Scrape table: We will fetch the scrape from this table with the given run ID. Required columns = [{', '.join(required_scrape_cols)}]",
        placeholder=TABLE_INPUT_PLACEHOLDER,
    )
    st.session_state.result_limit = st.number_input(
        "Result Limit (k): Cortex Search will retrive top k results for the queries",
        placeholder="10",
        step=1,
        value=10,
        min_value=1,
        max_value=100,
    )
    st.session_state.scrape_run_id = st.text_input(
        "Enter scrape Run ID: Unique ID to identify the existing scrape",
        placeholder="unique RUN_ID from the scrape table",
    )
    if st.session_state.scrape_fqn != "" and st.session_state.scrape_run_id != "":
        validate_scrape_fqn(must_exist=True)
        st.session_state.scrape_ready = True


def process_generate_scrape(db: str, schema: str) -> None:
    """Process scrape generation based on user input."""
    st.session_state.scrape_fqn = f"{db}.{schema}.css_scrape"
    st.session_state.result_limit = 10
    generate_and_store_scrape(session, root)
    st.session_state.scrape_ready = True


def initialize_session_state() -> None:
    session_defaults = {
        "scrape_ready": False,
        "scrape_fqn": "",
        "scrape_for_autotune_relevancy": False,
        "scrape_run_id": "",
        "result_limit": -1,
        "aggregate_metrics_display_df": "",
        "retrieval_metrics_per_query_df": "",
        "scrape_df": "",
        "colors": {},
        "rel_scores": {},
        "queryid_to_query": {},  # Keep the map in memory, to avoid redundant data storage of queries
        "prev_css_fqn": "",  # To avoid re-verification of unchanged values
        "prev_queryset_fqn": "",
        "prev_scrape_fqn": "",
        "prev_css_id_col": "",
        "css_id_col": "",
        "k_value": 5,
        "idcg_factor": 3.0,
        "relevancy_btn_clicked": False,
        "relevancy_provided": False,
        "run_evaluation": False,
        "run_autotuning": False,
        "autotuning_aggregate_metrics_display_df": "",
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.session_state.md5_hash = md5_hash_string(session)
    st.session_state.filter = {}  # customer can provide json filter , not used in v0
    st.session_state.additional_columns = []  # customer can provide more columns to retrieve, not used in v0


def display_header():
    st.title("Cortex Search Evaluation Support")
    st.markdown(
        """
        Tool to evaluate the quality of Cortex Search Service against a given of queries and relevancies

        **Note:** All table names are needed in a fully qualified name format, ie, `<db>.<schema>.<table>`
        """
    )
    st.header("Inputs", divider=True)


def collect_input_fields() -> None:
    st.session_state.css_fqn = st.text_input(
        "Enter the Cortex Search Service (CSS):",
        placeholder="in format db.schema.cortex_search_service",
    )
    st.session_state.queryset_fqn = st.text_input(
        f"Enter the Queries table: Required cols = [{', '.join(QUERIES_TABLE_COLUMNS)}]",
        placeholder=TABLE_INPUT_PLACEHOLDER,
    )
    st.markdown(
        """
            Do you have a relevancy table for your queries? 

            _The relevancy score for <query, document> pair will be used to compare our retrieval results_
            """
    )
    required_relevancy_cols = set(RELEVANCY_TABLE_COLUMNS)

    st.session_state.relevancy_fqn = ""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="rel_yes"):
            st.session_state.relevancy_provided = True
            st.session_state.relevancy_btn_clicked = True
            st.session_state.run_evaluation = False
            st.session_state.run_autotuning = False
    with col2:
        if st.button("No", key="rel_no"):
            st.session_state.relevancy_provided = False
            st.session_state.relevancy_btn_clicked = True
            st.session_state.run_evaluation = False
            st.session_state.run_autotuning = False

    if st.session_state.relevancy_provided:
        st.session_state.relevancy_fqn = st.text_input(
            f"Enter Relevancy table. This table which should have ground truth labels for the queries. Required columns = [{', '.join(required_relevancy_cols)}]",
            placeholder=TABLE_INPUT_PLACEHOLDER,
        )
        if st.session_state.relevancy_fqn == "":
            st.stop()

    if st.session_state.css_fqn == "" or st.session_state.queryset_fqn == "":
        st.stop()

    st.session_state.css_text_col = get_search_column().upper()
    required_relevancy_cols.add(st.session_state.css_text_col)
    if st.session_state.relevancy_provided:
        validate_table(
            fqn=st.session_state.relevancy_fqn,
            fqn_name="Relevancy table",
            session=session,
            must_exist=True,
            required_cols=required_relevancy_cols,
        )
        validate_doc_id_col(st.session_state.relevancy_fqn)


def all_required_inputs_provided():
    # Check base requirements
    base_requirements = [
        st.session_state.css_fqn,
        st.session_state.queryset_fqn,
        st.session_state.relevancy_btn_clicked,
    ]

    return all(base_requirements)


def inputs_have_changed():
    return any(
        [
            st.session_state.css_fqn != st.session_state.prev_css_fqn,
            st.session_state.queryset_fqn != st.session_state.prev_queryset_fqn,
            st.session_state.css_id_col != st.session_state.prev_css_id_col,
        ]
    )


def update_previous_inputs():
    st.session_state.prev_css_fqn = st.session_state.css_fqn
    st.session_state.prev_queryset_fqn = st.session_state.queryset_fqn
    st.session_state.prev_css_id_col = st.session_state.css_id_col


def validate_inputs():
    validate_css(
        st.session_state.css_fqn,
    )
    assert (
        len(st.session_state.css_text_col) > 0
    ), "Search column name for the Cortex Search Service is empty."

    validate_table(
        fqn=st.session_state.queryset_fqn,
        fqn_name="Queries table",
        session=session,
        must_exist=True,
        required_cols=set(QUERIES_TABLE_COLUMNS),
    )


def display_verified_inputs():
    st.write(f"Cortex Search Service = `{st.session_state.css_fqn}` ✅")
    st.write(f"Queries table = `{st.session_state.queryset_fqn}` ✅")


def extract_db_and_schema_from_fqn() -> List[str]:
    return st.session_state.css_fqn.split(".")[:2]

    # If order to allow mutliple scraping to eval loops gracefully


def refresh_state_to_pre_scrape():
    st.session_state.scrape_ready = False
    st.session_state.aggregate_metrics_display_df = ""
    st.session_state.autotuning_aggregate_metrics_display_df = ""
    st.session_state.retrieval_metrics_per_query_df = ""
    st.session_state.scrape_df = ""
    st.session_state.rel_scores = {}
    st.session_state.colors = {}


def process_scrape_workflow(db: str, schema: str, autotune: bool = False):
    process_eval_input(db, schema, autotune=autotune)


def process_eval_input(db, schema, autotune=False):
    required_relevancy_cols = set(RELEVANCY_TABLE_COLUMNS)
    required_relevancy_cols.add(st.session_state.css_text_col)

    result_fqn = f"{db}.{schema}.css_metrics"
    # run_comment = st.text_input("Add optional note", placeholder="stored as text")
    run_comment = "simple"
    if st.session_state.run_evaluation or not st.session_state.relevancy_provided:
        st.session_state.scrape_run_id = generate_runid()
        process_generate_scrape(db, schema)
    if st.session_state.scrape_ready or (
        st.session_state.run_autotuning or st.session_state.relevancy_provided
    ):
        st.success(
            f"""
            Scrape Stored in:
            - **Table:** {st.session_state.scrape_fqn}
            - **Run ID:** {st.session_state.scrape_run_id}"""
        )
        if not st.session_state.relevancy_provided:
            st.session_state.relevancy_fqn = (
                f"{db}.{schema}.llm_rel_{st.session_state.scrape_run_id}"
            )

        if autotune:
            run_autotuning(st.session_state.relevancy_fqn, result_fqn, run_comment)
        else:
            run_eval(st.session_state.relevancy_fqn, result_fqn, run_comment)


def eval_results_are_ready():
    return (
        st.session_state.aggregate_metrics_display_df != ""
        and st.session_state.retrieval_metrics_per_query_df != ""
        and st.session_state.scrape_df != ""
        and st.session_state.scrape_ready
    )


def autotuning_results_are_ready():
    return st.session_state.autotuning_aggregate_metrics_display_df != ""


def display_eval_results(autotune: bool = False):
    show_aggregate_results(autotune)
    if not autotune:
        show_query_level_results()

    st.session_state.run_evaluation = False
    st.session_state.run_autotuning = False


def extract_value_from_dict(debug_signals: str, key: str) -> Optional[Any]:
    try:
        signals = json.loads(debug_signals)
        return signals.get(key, None)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_relevancy_score(row: Dict[str, Any]) -> Union[int, str]:
    query_id = str(row[QUERY_ID])
    doc_id = str(row[DOC_ID])
    score = st.session_state.rel_scores.get(query_id, {}).get(doc_id, None)

    return int(score) if score is not None else "-"


def get_metrics_for_k(df: pd.DataFrame, k: str) -> Tuple[int]:
    """Fetches the metric values for the given k from the DataFrame."""
    metrics = (
        df[f"SDCG@{k}"].iloc[0],
        df[f"PRECISION@{k}"].iloc[0],
        df[f"HIT_RATE@{k}"].iloc[0],
    )
    return metrics


def display_english_metrics(df: pd.DataFrame, k_value: str) -> None:
    """Displays the metrics with explanations for  a chosen k."""
    sdcg_value, precision_value, hit_rate_value = get_metrics_for_k(df, k_value)
    st.write(
        f"**SDCG@{k_value} is {sdcg_value}** - Standardized Discounted Cumulative Gain (SDCG) measures the effectiveness of a search algorithm based on the positions of relevant results. Ranges from 0 (worst) to 1 (best)."
    )

    st.write(
        f"**Hit Rate@{k_value} is {hit_rate_value}** which means that in {hit_rate_value * 100}% of queries, at least one relevant result was present in the top {k_value} retrieved items."
    )

    st.write(
        f"**Precision@{k_value} is {precision_value}** which means that {precision_value * 100}% of the documents retrieved in the top {k_value} results were relevant."
    )


def show_aggregate_results(autotune: bool = False) -> None:
    # st.header("Evaluation Results", divider=True)
    st.header("Section: Static results", divider=True)

    aggregate_metrics_display_df = (
        st.session_state.autotuning_aggregate_metrics_display_df
        if autotune
        else st.session_state.aggregate_metrics_display_df
    )
    metrics_display_aggregate = aggregate_metrics_display_df.to_pandas()
    metrics_display_aggregate[COMMENT] = metrics_display_aggregate[RUN_METADATA].apply(
        extract_value_from_dict, key="Comment"
    )
    metrics_display_aggregate[TOTAL_QUERIES] = metrics_display_aggregate[
        RUN_METADATA
    ].apply(extract_value_from_dict, key="NumQueriesScraped")

    k_values = [
        limit for limit in VALID_LIMITS if limit <= st.session_state.result_limit
    ]
    metrics_data = {
        metric: [metrics_display_aggregate[f"{metric}@{k}"].values[0] for k in k_values]
        for metric in VALID_METRICS
    }
    st.subheader(
        """
        **Aggregate Retrieval Metrics**
        """,
        divider=True,
    )
    desired_order = [SDCG, HIT_RATE, PRECISION]
    metrics_df = pd.DataFrame(metrics_data, index=[f"k={k}" for k in k_values])
    metrics_df = metrics_df[desired_order]
    # You can't order columns as part of st.dataframe here, because we want metrics to be rows and not columns
    st.dataframe(metrics_df.T)

    st.subheader(
        """
        **Run Metadata**
        """,
        divider=True,
    )
    st.write(
        (metrics_display_aggregate[[RUN_ID, TOTAL_QUERIES, COMMENT, RUN_METADATA]]).T
    )

    st.header("Section: Interactive results at chosen k", divider=True)
    st.session_state.k_value = st.selectbox("Choose the value of k:", k_values, index=1)
    st.subheader("Aggregate Summary", divider=True)
    display_english_metrics(metrics_display_aggregate, str(st.session_state.k_value))


def show_qid_info(row, pq_df: pd.Series):
    scrape_pandas = st.session_state.scrape_df.to_pandas()
    scrape_pandas[QUERY] = scrape_pandas[QUERY_ID].map(
        st.session_state.queryid_to_query
    )
    scrape_pandas[FINAL_SCORE] = scrape_pandas[DEBUG_SIGNALS].apply(
        extract_value_from_dict, key="FinalScore"
    )
    scrape_pandas[RELEVANCY_SCORE] = scrape_pandas.apply(
        extract_relevancy_score, axis=1
    )

    data = row.selection
    if data["rows"]:
        idx = data["rows"][0]
        qid = pq_df.iloc[idx][QUERY_ID]
        qid_result_pandas = scrape_pandas[scrape_pandas[QUERY_ID] == qid].sort_values(
            by=[RANK]
        )
        qid_result = qid_result_pandas.style.apply(
            color_code_columns,
            axis=1,
            columns=qid_result_pandas.columns,
        )
        st.dataframe(
            qid_result,
            column_order=[
                RANK,
                QUERY,
                st.session_state.css_text_col,
                RELEVANCY_SCORE,
                FINAL_SCORE,
                QUERY_ID,
                DOC_ID,
            ],
            hide_index=True,
        )
        st.markdown(
            f"""
            **As marked by the given relevancy scores for (query, doc) pair** \n
            <span style="background-color:{lightgreen}; color:black">_GREEN_</span> - Most Relevant documents (scored 2 or 3) \n
            <span style="background-color:{lightyellow}; color:black">_YELLOW_</span> - Medium Relevancy documents (scored 1) \n
            <span style="background-color:{lightcoral}; color:black">_RED_</span> - Not Relevant documents (scored 0) \n
            _WHITE_ - Not scored by relevancy table\n
            """,
            unsafe_allow_html=True,
        )
    pq_scores_csv = scrape_pandas.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download all per-query scores (CSV)",
        data=pq_scores_csv,
        file_name="per_query_scores.csv",
        mime="text/csv",
    )


def show_query_level_results() -> None:
    metrics_per_query_pandas = (
        st.session_state.retrieval_metrics_per_query_df.to_pandas()
    )

    metrics_per_query_pandas[QUERY] = metrics_per_query_pandas[QUERY_ID].map(
        st.session_state.queryid_to_query
    )
    st.subheader("Per Query Metrics", divider=True)
    st.markdown("""
    You can select a row using the left-most index column to learn more about a specific query. 
    
    You could also click on any of the columns to sort on that column
    """)
    pq_series = extract_and_sort_metrics(metrics_per_query_pandas)
    k_value = str(st.session_state.k_value)

    column_configs = {}
    for column in pq_series.columns:
        column_configs[column] = st.column_config.Column(
            column,
            width="small",
            required=True,
        )
    selected_row = st.dataframe(
        pq_series,
        width=150000,
        hide_index=True,
        column_config=column_configs,
        on_select="rerun",
        selection_mode=["single-row"],
        column_order=[
            QUERY,
            QUERY_ID,
            f"{SDCG}@{k_value}",
            f"{HIT_RATE}@{k_value}",
            f"{PRECISION}@{k_value}",
        ],
    )
    show_qid_info(selected_row, pq_series)


@st.cache_resource
def get_session() -> Session:
    try:
        # For SiS case
        return get_active_session()
    except SnowparkSessionException:
        # For local dev
        return Session.builder.configs(
            {
                "user": os.environ["USER"],
                "account": "sfengineering-mldataconnection",
                "warehouse": "SEARCH_L",
                "role": "CORTEX_SEARCH_TEAM",
                "database": "CORTEX_SEARCH_DB",
                "schema": os.environ["USER"],
                "authenticator": "externalbrowser",
            }
        ).create()


def main():
    initialize_session_state()
    display_header()
    collect_input_fields()

    if not all_required_inputs_provided():
        st.stop()

    if inputs_have_changed():
        update_previous_inputs()
        validate_inputs()
    else:
        display_verified_inputs()

    db, schema = extract_db_and_schema_from_fqn()

    st.markdown(
        """
        Would you like to run Evaluation or Autotuning
        """
    )
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Evaluation"):
            st.session_state.run_evaluation = True
            st.session_state.run_autotuning = False
            refresh_state_to_pre_scrape()

    with col2:
        if st.button("Run Autotuning"):
            st.session_state.run_autotuning = True
            st.session_state.run_evaluation = False
            refresh_state_to_pre_scrape()

    if st.session_state.run_evaluation:
        process_eval_input(db, schema, autotune=False)

    elif st.session_state.run_autotuning:
        process_eval_input(db, schema, autotune=True)

    if eval_results_are_ready():
        display_eval_results()
    if autotuning_results_are_ready():
        display_eval_results(autotune=True)


if __name__ == "__main__":
    session = get_session()
    root = Root(session)
    main()
