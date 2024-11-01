# Standard library imports
import hashlib
import json
import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import os

# Third-party library imports
import pandas as pd
import streamlit as st

# Snowflake-specific imports
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSessionException
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.functions import col, max as spmax, udf
from snowflake.snowpark.session import Session
from snowflake.snowpark.table import Table
from snowflake.snowpark.dataframe import DataFrame
from snowflake.snowpark.types import StringType
from snowflake.snowpark.row import Row
from snowflake.core import Root


# --- Constants ---
RUN_ID = "RUN_ID"
DOC_ID = "DOC_ID"
RANK = "RANK"
DEBUG_SIGNALS = "DEBUG_SIGNALS"
QUERY_ID = "QUERY_ID"
QUERY = "QUERY"
CSS = "CORTEX SEARCH SERVICE"
DEBUG_PER_RESULT = "@DEBUG_PER_RESULT"
HIT_RATE = "HIT_RATE"
NDCG = "NDCG"
RECALL = "RECALL"
PRECISION = "PRECISION"
VALID_METRICS = (HIT_RATE, NDCG, RECALL, PRECISION)
VALID_LIMITS = (1, 3, 5, 10, 20, 50, 100)
RELEVANCY = "RELEVANCY"
FINAL_SCORE = "FINAL_SCORE"
RELEVANCY_SCORE = "RELEVANCY_SCORE"
TOTAL_QUERIES = "TOTAL_QUERIES"
COMMENT = "COMMENT"
RUN_METADATA = "RUN_METADATA"
TABLE_INPUT_PLACEHOLDER = "in format db.schema.table"
QUERIES_TABLE_COLUMNS = (QUERY,)  # ',' allows casting this into a set later
RELEVANCY_TABLE_COLUMNS = (
    QUERY,
    DOC_ID,
    RELEVANCY,
)
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
    NDCG,
    PRECISION,
    RECALL,
)
lightgreen = "lightgreen"
lightyellow = "lightyellow"
lightcoral = "lightcoral"


# --- Utility Functions ---
def hit_rate(results: List[str], golden_to_score: Dict[str, Dict[str, int]]) -> int:
    """Calculate hit rate for search results."""
    for result in results:
        if result in golden_to_score:
            return 1 if golden_to_score[result]["score"] > 0 else 0
    return 0


def ndcg(results: List[str], golden_to_score: Dict[str, Dict[str, int]]) -> float:
    """Calculate NDCG for search results."""
    k = min(len(results), len(golden_to_score))
    ideal_golden_results = sorted(golden_to_score.items(), key=lambda x: x[1]["rank"])[
        :k
    ]
    ideal = _dcg([doc[0] for doc in ideal_golden_results], golden_to_score)

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


def recall(results: List[str], golden_to_score: Dict[str, Dict[str, int]]) -> float:
    """Calculate recall for search results."""
    if not golden_to_score:
        return float("nan")

    return count_relevant_results(results, golden_to_score) / len(golden_to_score)


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
        metrics (dict): Metric functions (precision, recall, etc.) to apply.
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

    if row[QUERY_ID] in st.session_state.colors:
        doc_colors = st.session_state.colors[row[QUERY_ID]]

        if row[DOC_ID] in doc_colors:
            color = doc_colors[row[DOC_ID]]
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
        return ""


def validate_css_id_col() -> bool:
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
        css_columns = set(col.lower() for col in result[0][7].split(","))
        return st.session_state.css_id_col.lower() in css_columns
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return False


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


def prepare_query_table(session: Session) -> Table:
    """Prepare the query table and ensure QUERY_ID column exists."""
    query_table = session.table(st.session_state.queryset_fqn)

    if QUERY_ID not in query_table.columns:
        query_table = query_table.withColumn(
            QUERY_ID, st.session_state.md5_hash(query_table[QUERY])
        )  # Generate QUERY_ID if missing

    return query_table


def perform_scrape(query_table: Table, root: Root) -> List[Dict[str, Any]]:
    """Perform scraping for each query and return the results."""
    css_db, css_schema, css_service = st.session_state.css_fqn.split(".")
    svc = root.databases[css_db].schemas[css_schema].cortex_search_services[css_service]
    scrape_out = []
    all_queries = query_table.collect()
    validate_queryset(all_queries)
    total_rows = len(all_queries)
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Scraping in progress.. (this may take a while)")

    for i, row in enumerate(all_queries):
        progress_percentage = int((i + 1) / total_rows * 100)
        progress_bar.progress(progress_percentage)
        query, query_id = row[QUERY], row[QUERY_ID]

        # Perform the search request
        response = svc.search(
            query,
            limit=st.session_state.result_limit,
            columns=[st.session_state.css_text_col, st.session_state.css_id_col]
            + st.session_state.additional_columns,
            filter=st.session_state.filter,
            # Todo(amehta): input experimental args and use them here
            # debug set to True to display debug signals
            experimental={"debug": True},
        )
        # Append each result to the scrape output
        for rank, result in enumerate(response.results):
            scrape_out.append(
                {
                    RUN_ID: st.session_state.scrape_run_id,
                    QUERY_ID: query_id,
                    DOC_ID: result[st.session_state.css_id_col],
                    RANK: rank + 1,
                    DEBUG_SIGNALS: result[DEBUG_PER_RESULT],
                    st.session_state.css_text_col.upper(): result[
                        st.session_state.css_text_col
                    ],  # Todo(amehta): Handle markdown if necessary
                }
            )

    return scrape_out


def store_scrape_results(scrape_out: List[Dict[str, Any]], session: Session) -> None:
    """Store scrape results in the Snowflake table and display progress."""
    scrape_df = session.create_dataframe(scrape_out)
    scrape_df.write.mode("append").save_as_table(st.session_state.scrape_fqn)

    duration = datetime.now() - st.session_state.start_time
    st.success(
        f"Finished scrape in {round(duration.total_seconds(),1)} seconds, stored in `{st.session_state.scrape_fqn}` with Run ID = {st.session_state.scrape_run_id}"
    )


def generate_and_store_scrape(session: Session, root: Root) -> None:
    """Generate and store scrape results."""
    st.session_state.start_time = datetime.now()

    query_table = prepare_query_table(session)
    scrape_out = perform_scrape(query_table, root)

    store_scrape_results(scrape_out, session)


def run_eval(relevancy_fqn: str, result_fqn: str, run_comment: str) -> None:
    """Run evaluation on scraped results against golden standards."""
    start_time = datetime.now()
    query_table, scrape_df = initialize_tables(session)

    validate_scrape(scrape_df)

    st.session_state.result_limit = get_result_limit(scrape_df)

    relevancy_table = prepare_relevancy_table(relevancy_fqn, session)

    raw_goldens = extract_and_dedupe_goldens(relevancy_table)
    goldens = prepare_golden_scores(raw_goldens)

    retrieval_metrics = evaluate_queries(query_table, scrape_df, goldens)
    persist_metrics(
        retrieval_metrics, relevancy_fqn, result_fqn, run_comment, scrape_df
    )

    duration = datetime.now() - start_time
    st.session_state.scrape_df = scrape_df
    st.session_state.queryid_to_query = {
        row[QUERY_ID]: row[QUERY] for row in query_table.collect()
    }

    st.success(
        f"Evaluation finished in {round(duration.total_seconds(), 1)} seconds\nEval metrics are added to `{result_fqn}`"
    )


def initialize_tables(session: Session) -> Tuple[Table, DataFrame]:
    """Initialize query and scrape tables."""
    query_table = session.table(st.session_state.queryset_fqn)

    if QUERY_ID not in query_table.columns:
        query_table = query_table.withColumn(
            QUERY_ID, st.session_state.md5_hash(query_table[QUERY])
        )

    scrape_table = session.table(st.session_state.scrape_fqn)
    scrape_df = scrape_table.filter(col(RUN_ID) == st.session_state.scrape_run_id)

    return query_table, scrape_df


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


def prepare_relevancy_table(relevancy_fqn: str, session: Session) -> Table:
    """Prepare the relevancy table, ensuring QUERY_ID is present."""
    relevancy_table = session.table(relevancy_fqn)
    if QUERY_ID not in relevancy_table.columns:
        relevancy_table = relevancy_table.withColumn(
            QUERY_ID, st.session_state.md5_hash(relevancy_table[QUERY])
        )
    return relevancy_table


def extract_and_dedupe_goldens(
    relevancy_table: Table,
) -> Dict[str, List[Tuple[str, int]]]:
    """Extract golden scores from the relevancy table."""
    raw_goldens: Dict[str, List[Tuple[str, int]]] = {}
    relevance_color_mapping = {
        0: lightcoral,
        1: lightyellow,
        2: lightgreen,
        3: lightgreen,
    }

    for row in relevancy_table.collect():
        rel_score = int(row[RELEVANCY])
        query_id, doc_id = row[QUERY_ID], row[DOC_ID]

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
    query_table: Table,
    scrape_df: DataFrame,
    goldens: Dict[str, Dict[str, Dict[str, int]]],
) -> List[Dict[str, Any]]:
    """Evaluate all queries against the scraped results."""
    retrieval_metrics: List[Dict[str, Any]] = []
    progress_bar = st.progress(0)
    total_rows = len(query_table.collect())
    status_text = st.empty()
    status_text.text("Evaluation in progress.. (this may take a while)")

    for i, row in enumerate(query_table.collect()):
        progress_percentage = (i + 1) / total_rows
        progress_bar.progress(progress_percentage)

        query_id = row[QUERY_ID]
        scrape_for_query_df = scrape_df.filter(col(QUERY_ID) == query_id)

        # Dedup doc_ids in result while persisting order
        scraped_results = list(
            dict.fromkeys(r[DOC_ID] for r in scrape_for_query_df.collect())
        )

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
                    NDCG: calculate_metrics(
                        scraped_results, golden_to_score, {NDCG: ndcg}, VALID_LIMITS
                    )[NDCG],
                    PRECISION: calculate_metrics(
                        scraped_results,
                        golden_to_score,
                        {PRECISION: precision},
                        VALID_LIMITS,
                    )[PRECISION],
                    RECALL: calculate_metrics(
                        scraped_results, golden_to_score, {RECALL: recall}, VALID_LIMITS
                    )[RECALL],
                }
            )

    return retrieval_metrics


def calculate_aggregate_metrics(
    retrieval_metrics: List[Dict[str, Any]],
    relevancy_fqn: str,
    run_comment: str,
    scrape_df: DataFrame,
) -> Dict[str, Any]:
    """Calculate aggregate metrics from retrieval metrics."""
    return {
        RUN_ID: st.session_state.scrape_run_id,
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
        NDCG: calculate_average_metrics(retrieval_metrics, NDCG),
        PRECISION: calculate_average_metrics(retrieval_metrics, PRECISION),
        RECALL: calculate_average_metrics(retrieval_metrics, RECALL),
    }


def create_display_metrics(
    aggregate_metrics: Dict[str, Any], retrieval_metrics: List[Dict[str, Any]]
) -> DataFrame:
    """Create display metrics DataFrame for user interface."""
    display_data = {
        RUN_ID: st.session_state.scrape_run_id,
        RUN_METADATA: aggregate_metrics[RUN_METADATA],
        **compute_display_metrics(retrieval_metrics, st.session_state.result_limit),
    }
    return session.create_dataframe([display_data])


def persist_metrics(
    retrieval_metrics: List[Dict[str, Any]],
    relevancy_fqn: str,
    result_fqn: str,
    run_comment: str,
    scrape_df: DataFrame,
) -> None:
    """Persist the metrics in the database."""

    # Create DataFrame for per-query metrics and save it
    st.session_state.retrieval_metrics_per_query_df = session.create_dataframe(
        retrieval_metrics
    )
    per_query_fqn = f"{result_fqn}_PERQUERY"
    st.session_state.retrieval_metrics_per_query_df.write.mode("append").save_as_table(
        per_query_fqn
    )

    # Create DataFrame for aggregate metrics and save it
    aggregate_metrics = calculate_aggregate_metrics(
        retrieval_metrics, relevancy_fqn, run_comment, scrape_df
    )
    retrieval_metrics_df = session.create_dataframe([aggregate_metrics])
    retrieval_metrics_df.write.mode("append").save_as_table(result_fqn)

    # Display aggregate metrics to the user
    st.session_state.aggregate_metrics_display_df = create_display_metrics(
        aggregate_metrics, retrieval_metrics
    )


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
    st.session_state.scrape_run_id = st.text_input(
        "Enter scrape Run ID: Unique ID to identify the existing scrape",
        placeholder="unique RUN_ID from the scrape table",
    )
    if st.session_state.scrape_fqn != "" and st.session_state.scrape_run_id != "":
        validate_scrape_fqn(must_exist=True)
        st.session_state.scrape_ready = True


def process_generate_scrape() -> None:
    """Process scrape generation based on user input."""
    st.session_state.scrape_fqn = st.text_input(
        "Choose a name for your Scrape table: Generated Scrape will be stored in this table",
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
    if (
        st.button("Generate Scrape") and st.session_state.scrape_fqn != ""
    ):  # We need this button, because we're autofilling the scrape_fqn value
        st.session_state.scrape_run_id = generate_runid()
        if check_table_exists(st.session_state.scrape_fqn):
            st.markdown(
                f"""
                <div style='background-color: yellow; padding: 10px;'>
                    <strong>Warning!</strong> Scrape Table `{st.session_state.scrape_fqn}` exists. Will append the generated scrape.
                </div>
                """,
                unsafe_allow_html=True,
            )
            validate_scrape_fqn(must_exist=True)
        else:
            validate_scrape_fqn(must_exist=False)

        generate_and_store_scrape(session, root)
        st.session_state.scrape_ready = True


def initialize_session_state() -> None:
    session_defaults = {
        "provide_scrape_clicked": False,
        "generate_scrape_clicked": False,
        "scrape_ready": False,
        "scrape_fqn": "",
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
        "k_value": 5,
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.session_state.md5_hash = md5_hash_string(session)
    st.session_state.filter = {}  # customer can provide json filter , not used in v0
    st.session_state.additional_columns = []  # customer can provide more columns to retrieve, not used in v0


def display_header():
    st.title("Cortex Search Evaluation Support")
    st.write(
        """
        Tool to evaluate the quality of Cortex Search Service against a set of queries
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
    st.session_state.css_id_col = st.text_input(
        "Enter the ID column name used in Cortex Search Service: All the IDs must be unique"
    )


def all_required_inputs_provided():
    return all(
        [
            st.session_state.css_fqn,
            st.session_state.queryset_fqn,
            st.session_state.css_id_col,
        ]
    )


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
    assert validate_css_id_col(), f"ID column = {st.session_state.css_id_col} does not exist for the Cortex Search Service = {st.session_state.css_fqn}"
    st.session_state.css_text_col = get_search_column().upper()
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
def refresh_state_to_pre_eval():
    st.session_state.scrape_ready = False
    st.session_state.aggregate_metrics_display_df = ""
    st.session_state.retrieval_metrics_per_query_df = ""
    st.session_state.scrape_df = ""
    st.session_state.rel_scores = {}
    st.session_state.colors = {}


def handle_scrape_input(db: str, schema: str):
    st.markdown(
        """
        Do you have a scrape?
        A scrape table contains the Cortex Search results for all the queries provided in the Query Table
        
        _Consider clicking No if you're a first time user_
        """
    )
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes"):
            st.session_state.provide_scrape_clicked = True
            st.session_state.generate_scrape_clicked = False
            refresh_state_to_pre_eval()

    with col2:
        if st.button("No"):
            st.session_state.generate_scrape_clicked = True
            st.session_state.provide_scrape_clicked = False
            refresh_state_to_pre_eval()

    if (
        st.session_state.provide_scrape_clicked
        or st.session_state.generate_scrape_clicked
    ):
        process_scrape_workflow(db, schema)


def process_scrape_workflow(db: str, schema: str):
    if st.session_state.provide_scrape_clicked and not st.session_state.scrape_ready:
        process_obtain_scrape()

    elif st.session_state.generate_scrape_clicked and not st.session_state.scrape_ready:
        process_generate_scrape()

    if st.session_state.scrape_ready:
        process_eval_input(db, schema)


def process_eval_input(db, schema):
    relevancy_fqn = st.text_input(
        f"Enter Relevancy table. This table which should have ground truth labels for the queries. Required columns = [{', '.join(RELEVANCY_TABLE_COLUMNS)}]",
        placeholder=TABLE_INPUT_PLACEHOLDER,
    )
    result_fqn = st.text_input(
        "Choose a name for your Metrics Table: Generated Evaluation Metrics will be appended in this table",
        placeholder=TABLE_INPUT_PLACEHOLDER,
        value=f"{db}.{schema}.CORTEX_SEARCH_METRICS",
    )
    run_comment = st.text_input("Add optional note", placeholder="stored as text")

    if st.button("Run Eval"):
        if (
            relevancy_fqn == ""
            or result_fqn == ""
            or st.session_state.scrape_run_id == ""
        ):
            st.stop()
        validate_table(
            fqn=relevancy_fqn,
            fqn_name="Relevancy table",
            session=session,
            must_exist=True,
            required_cols=set(RELEVANCY_TABLE_COLUMNS),
        )
        if check_table_exists(result_fqn):
            st.markdown(
                f"""
                <div style='background-color: yellow; padding: 10px;'>
                    <strong>Warning!</strong> Metrics Table {result_fqn} exists. Will append the generated metrics.
                </div>
                """,
                unsafe_allow_html=True,
            )
            validate_table(
                fqn=result_fqn,
                fqn_name="Metrics table",
                session=session,
                must_exist=True,
                required_cols=set(METRICS_TABLE_COLUMNS),
            )
        else:
            validate_table(
                fqn=result_fqn,
                fqn_name="Metrics table",
                session=session,
                must_exist=False,
            )

        run_eval(relevancy_fqn, result_fqn, run_comment)


def eval_results_are_ready():
    return (
        st.session_state.aggregate_metrics_display_df != ""
        and st.session_state.retrieval_metrics_per_query_df != ""
        and st.session_state.scrape_df != ""
        and st.session_state.scrape_ready
    )


def display_eval_results():
    show_aggregate_results()
    show_query_level_results()


def extract_value_from_dict(debug_signals: str, key: str) -> Optional[Any]:
    try:
        signals = json.loads(debug_signals)
        return signals.get(key, None)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_relevancy_score(row: Dict[str, Any]) -> Union[int, str]:
    query_id = row[QUERY_ID]
    doc_id = row[DOC_ID]
    score = st.session_state.rel_scores.get(query_id, {}).get(doc_id, None)

    return int(score) if score is not None else "-"


def get_metrics_for_k(df: pd.DataFrame, k: str) -> Tuple[int]:
    """Fetches the metric values for the given k from the DataFrame."""
    metrics = (
        df[f"NDCG@{k}"].iloc[0],
        df[f"RECALL@{k}"].iloc[0],
        df[f"PRECISION@{k}"].iloc[0],
        df[f"HIT_RATE@{k}"].iloc[0],
    )
    return metrics


def display_english_metrics(df: pd.DataFrame, k_value: str) -> None:
    """Displays the metrics with explanations."""
    ndcg_value, recall_value, precision_value, hit_rate_value = get_metrics_for_k(
        df, k_value
    )

    st.write(
        f"**Recall@{k_value} is {recall_value}** which means that on average for all queries, {recall_value*100}% of the relevant results were successfully retrieved in the top {k_value} results."
    )

    st.write(
        f"**Precision@{k_value} is {precision_value}** which means that {precision_value*100}% of the documents retrieved in the top {k_value} results were relevant."
    )

    st.write(
        f"**Hit Rate@{k_value} is {hit_rate_value}** which means that in {hit_rate_value*100}% of queries, at least one relevant result was present in the top {k_value} retrieved items."
    )

    st.write(
        f"**NDCG@{k_value} is {ndcg_value}** - Normalized Discounted Cumulative Gain (NDCG) measures the effectiveness of a search algorithm based on the positions of relevant results. Ranges from 0 (worst) to 1 (best)."
    )


def show_aggregate_results() -> None:
    st.header("Evaluation Results", divider=True)
    st.subheader("Section: Overall results", divider=True)

    metrics_display_aggregate = (
        st.session_state.aggregate_metrics_display_df.to_pandas()
    )
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
    st.markdown("""
        **Aggregate Retrieval Metrics**
        """)
    metrics_df = pd.DataFrame(metrics_data, index=[f"k={k}" for k in k_values])
    st.dataframe(metrics_df.T)

    st.divider()

    st.markdown("""
        **Run Metadata**
        """)
    st.write(
        (metrics_display_aggregate[[RUN_ID, TOTAL_QUERIES, COMMENT, RUN_METADATA]]).T
    )
    st.subheader("Evaluation Metrics at chosen k", divider=True)
    st.session_state.k_value = st.selectbox("Choose the value of k:", k_values, index=1)
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
            columns=[QUERY, QUERY_ID, st.session_state.css_text_col, FINAL_SCORE],
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
    st.divider()

    st.markdown("""
    **Per Query Metrics:**

    You can select a row using the left-most column to learn more about a specific query
    """)
    pq_series = extract_and_sort_metrics(metrics_per_query_pandas)
    remaining_columns = set(pq_series.columns) - set([QUERY, QUERY_ID])

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
        column_order=[QUERY, QUERY_ID] + list(remaining_columns),
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

    handle_scrape_input(db, schema)

    if eval_results_are_ready():
        display_eval_results()


if __name__ == "__main__":
    session = get_session()
    root = Root(session)
    main()
