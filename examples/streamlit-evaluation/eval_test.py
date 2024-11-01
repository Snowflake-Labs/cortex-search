import unittest
import math
from eval import (
    hit_rate,
    ndcg,
    precision,
    recall,
    _dcg,
    generate_and_store_scrape,
    prepare_query_table,
    perform_scrape,
    store_scrape_results,
    initialize_tables,
    validate_scrape,
    get_result_limit,
    prepare_relevancy_table,
    extract_and_dedupe_goldens,
    prepare_golden_scores,
    evaluate_queries,
    QUERY_ID,
    QUERY,
    HIT_RATE,
    RELEVANCY,
    RUN_ID,
    DOC_ID,
    NDCG,
    PRECISION,
    RECALL,
    DEBUG_PER_RESULT,
)
from unittest.mock import patch, MagicMock
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.snowpark import Table, DataFrame


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Setup some reusable test data
        self.results = ["doc1", "doc2", "doc3", "doc4"]
        self.golden_to_score = {
            "doc1": {"score": 3, "rank": 1},
            "doc2": {"score": 2, "rank": 2},
            "doc5": {"score": 1, "rank": 3},
            "doc6": {"score": 0, "rank": 4},
        }
        self.empty_results = []
        self.empty_golden = {}

    def test_hit_rate(self):
        # Expected to return 1 because doc1 is in golden_to_score and has score > 0
        self.assertEqual(hit_rate(self.results, self.golden_to_score), 1)

        # Expected to return 0 because no documents in results have score > 0
        self.assertEqual(hit_rate(["doc4"], self.golden_to_score), 0)

        # Test with empty results
        self.assertEqual(hit_rate(self.empty_results, self.golden_to_score), 0)

    def test_ndcg(self):
        # Calculate DCG and NDCG values for given results and golden data
        expected_ndcg = _dcg(self.results, self.golden_to_score) / _dcg(
            ["doc1", "doc2", "doc5"], self.golden_to_score
        )
        self.assertAlmostEqual(
            ndcg(self.results, self.golden_to_score), expected_ndcg, places=5
        )

        # Test NDCG with empty results
        self.assertEqual(ndcg(self.empty_results, self.golden_to_score), 0.0)

        # Test NDCG with empty golden set
        self.assertEqual(ndcg(self.results, self.empty_golden), 0.0)

    def test_precision(self):
        # 2 relevant documents (doc1, doc2), out of 4 total -> precision = 2/4 = 0.5
        self.assertAlmostEqual(precision(self.results, self.golden_to_score), 0.5)

        # No relevant documents, precision should be 0
        self.assertEqual(precision(["doc4"], self.golden_to_score), 0.0)

        # Test precision with empty results
        self.assertEqual(precision(self.empty_results, self.golden_to_score), 0.0)

    def test_recall(self):
        # 2 relevant documents (doc1, doc2) out of 4 golden -> recall = 2/4 = 0.5
        self.assertAlmostEqual(recall(self.results, self.golden_to_score), 0.5)

        # Test recall with no golden documents -> undefined recall, should return NaN
        self.assertTrue(math.isnan(recall(self.results, self.empty_golden)))

        # Test recall with no results
        self.assertEqual(recall(self.empty_results, self.golden_to_score), 0.0)

    def test_dcg(self):
        # Test DCG for correct calculation
        expected_dcg = 3 / math.log2(2) + 2 / math.log2(3)  # score of doc1 and doc2
        self.assertAlmostEqual(
            _dcg(self.results, self.golden_to_score), expected_dcg, places=5
        )

        # Test DCG with empty results
        self.assertEqual(_dcg(self.empty_results, self.golden_to_score), 0.0)


class TestScrapeFlow(unittest.TestCase):
    @patch("eval.get_active_session")
    @patch("eval.Root")
    @patch("streamlit.session_state", new_callable=MagicMock)
    @patch("eval.prepare_query_table")
    @patch("eval.perform_scrape")
    @patch("eval.store_scrape_results")
    def test_generate_and_store_scrape(
        self,
        mock_store_scrape_results,
        mock_perform_scrape,
        mock_prepare_query_table,
        mock_session_state,
        mock_root,
        mock_get_active_session,
    ):
        # Mock session state properties
        mock_session_state.start_time = datetime(2024, 1, 1)

        # Mock return values
        mock_query_table = MagicMock(name="query_table")
        mock_prepare_query_table.return_value = mock_query_table
        mock_scrape_out = [{"example_key": "example_value"}]
        mock_perform_scrape.return_value = mock_scrape_out

        # Mock session and root
        mock_session = mock_get_active_session.return_value
        mock_root_instance = mock_root.return_value

        # Call function
        generate_and_store_scrape(mock_session, mock_root_instance)

        # Assertions
        mock_prepare_query_table.assert_called_once()
        mock_perform_scrape.assert_called_once_with(
            mock_query_table, mock_root_instance
        )
        mock_store_scrape_results.assert_called_once_with(mock_scrape_out, mock_session)

    @patch("eval.get_active_session")  # Mock get_active_session
    @patch("eval.get_session")  # Mock get_session
    @patch("eval.Root")  # Mock Root
    @patch("streamlit.session_state", new_callable=MagicMock)  # Mock st.session_state
    def test_prepare_query_table_creates_query_id_if_missing(
        self, mock_session_state, mock_root, mock_get_session, mock_get_active_session
    ):
        # Define mock properties for session state
        mock_session_state.queryset_fqn = "mock_fqn"
        mock_session_state.md5_hash.return_value = "mock_query_id"

        # Mock the session object returned by get_session
        mock_session = MagicMock(spec=Session)
        mock_get_session.return_value = mock_session

        # Mock the table method on the session
        mock_query_table = MagicMock()
        mock_query_table.columns = [QUERY]  # Simulate missing QUERY_ID initially
        mock_query_table.withColumn.return_value = mock_query_table

        # Setup the session.table mock to return mock_query_table
        mock_session.table.return_value = mock_query_table

        # Call the function
        result = prepare_query_table(mock_session)  # Pass the mock session

        # Assertions
        mock_session.table.assert_called_once_with(
            "mock_fqn"
        )  # Check if session.table was called correctly
        mock_query_table.withColumn.assert_called_once()  # Check if withColumn was called
        self.assertEqual(
            result, mock_query_table
        )  # Ensure the result is the mock query_table

    @patch("eval.get_active_session")
    @patch("eval.Root")
    @patch("eval.st.session_state", new_callable=MagicMock)
    @patch("eval.st.progress")
    @patch("eval.st.empty")
    def test_perform_scrape(
        self,
        mock_empty,
        mock_progress,
        mock_session_state,
        mock_root,
        mock_get_active_session,
    ):
        # Mock session state properties
        mock_session_state.css_fqn = "mock_db.mock_schema.mock_service"
        mock_session_state.result_limit = 10
        mock_session_state.css_text_col = "text"
        mock_session_state.css_id_col = DOC_ID
        mock_session_state.additional_columns = []
        mock_session_state.filter = None
        mock_session_state.scrape_run_id = "mock_run_id"

        # Mocking session and root
        mock_root_instance = mock_root.return_value

        # Mock service setup
        mock_css_db, mock_css_schema, mock_css_service = (
            mock_session_state.css_fqn.split(".")
        )
        mock_svc = MagicMock()

        # Ensure cortex_search_services[service] returns mock_svc
        mock_root_instance.databases[mock_css_db].schemas[
            mock_css_schema
        ].cortex_search_services = {mock_css_service: mock_svc}

        # Mock the search result
        mock_search_result = MagicMock()
        mock_search_result.results = [
            {
                mock_session_state.css_id_col: "123",
                mock_session_state.css_text_col: "sample text",
                DEBUG_PER_RESULT: "signal_data",
            }
        ]
        mock_svc.search.return_value = mock_search_result

        # Mock query table collection
        mock_query_table = MagicMock()
        mock_query_table.collect.return_value = [{QUERY: "test query", QUERY_ID: "1"}]
        mock_progress().progress.side_effect = lambda x: None
        mock_empty().text.side_effect = lambda x: None

        scrape_out = perform_scrape(mock_query_table, mock_root_instance)
        print(scrape_out)

        mock_svc.search.assert_called_once()  # Validate search was called
        self.assertEqual(len(scrape_out), 1)  # Confirm one result was returned
        self.assertEqual(scrape_out[0][QUERY_ID], "1")  # Check QUERY_ID matches
        self.assertEqual(scrape_out[0][DOC_ID], "123")  # Check DOC_ID matches

    @patch("eval.get_active_session")
    @patch("eval.st.session_state", new_callable=MagicMock)
    def test_store_scrape_results(self, mock_session_state, mock_get_active_session):
        # Mock session state properties
        mock_session_state.scrape_fqn = "mock_db.mock_schema.mock_scrape_table"
        mock_session_state.start_time = datetime.now()
        mock_session_state.scrape_run_id = "mock_run_id"

        # Mock session and DataFrame
        mock_session = mock_get_active_session.return_value
        mock_dataframe = MagicMock()
        mock_session.create_dataframe.return_value = mock_dataframe
        mock_write = MagicMock()
        mock_dataframe.write.mode.return_value = mock_write
        mock_write.save_as_table.return_value = None

        # Sample scrape output
        scrape_out = [{RUN_ID: "mock_run_id", QUERY_ID: "1", DOC_ID: "123", "RANK": 1}]

        # Call function
        store_scrape_results(scrape_out, mock_session)

        # Assertions
        mock_session.create_dataframe.assert_called_once_with(scrape_out)
        mock_dataframe.write.mode.assert_called_once_with("append")
        mock_dataframe.write.mode().save_as_table.assert_called_once_with(
            "mock_db.mock_schema.mock_scrape_table"
        )


class TestEvalFlow(unittest.TestCase):
    @patch("eval.get_session")
    @patch("streamlit.session_state", new_callable=MagicMock)
    def test_initialize_tables(self, mock_session_state, mock_get_session):
        mock_session_state.queryset_fqn = "mock_queryset_fqn"
        mock_session_state.scrape_fqn = "mock_scrape_fqn"
        mock_session_state.scrape_run_id = "mock_run_id"

        mock_session = MagicMock(spec=Session)
        mock_query_table = MagicMock(spec=Table)
        mock_scrape_table = MagicMock(spec=Table)
        mock_scrape_df = MagicMock(spec=DataFrame)

        mock_get_session.return_value = mock_session
        mock_session.table.side_effect = [mock_query_table, mock_scrape_table]

        mock_query_table.columns = [QUERY]
        mock_query_table.withColumn.return_value = mock_query_table
        mock_scrape_table.filter.return_value = mock_scrape_df
        mock_session.table.return_value = mock_query_table

        query_table, scrape_df = initialize_tables(mock_session)

        mock_session.table.assert_any_call("mock_queryset_fqn")
        mock_session.table.assert_any_call("mock_scrape_fqn")
        mock_query_table.withColumn.assert_called_once_with(
            QUERY_ID, mock_session_state.md5_hash(mock_query_table[QUERY])
        )
        mock_scrape_table.filter.assert_called_once()
        self.assertEqual(query_table, mock_query_table)
        self.assertEqual(scrape_df, mock_scrape_df)

    @patch("streamlit.session_state", new_callable=MagicMock)
    def test_validate_scrape(self, mock_session_state):
        mock_scrape_df = MagicMock(spec=DataFrame)
        mock_scrape_df.count.return_value = 1

        validate_scrape(mock_scrape_df)

        mock_scrape_df.count.return_value = 0
        with self.assertRaises(
            AssertionError,
            msg="Scrape is empty! Recheck the Run ID or the Scrape table.",
        ):
            validate_scrape(mock_scrape_df)

    @patch("eval.spmax")
    def test_get_result_limit(self, mock_spmax):
        mock_scrape_df = MagicMock(spec=DataFrame)
        mock_spmax.return_value = MagicMock()
        mock_scrape_df.select.return_value.collect.return_value = [[5]]

        result = get_result_limit(mock_scrape_df)
        self.assertEqual(result, 5)

    @patch("streamlit.session_state", new_callable=MagicMock)
    @patch("eval.get_session")
    def test_prepare_relevancy_table(self, mock_get_session, mock_session_state):
        mock_session_state.relevancy_fqn = "mock_relevancy_fqn"
        mock_session = MagicMock(spec=Session)
        mock_relevancy_table = MagicMock(spec=Table)
        mock_modified_relevancy_table = MagicMock(spec=Table)  # Mock for modified table

        mock_get_session.return_value = mock_session
        mock_session.table.return_value = mock_relevancy_table
        mock_relevancy_table.columns = [QUERY, DOC_ID]
        mock_relevancy_table.withColumn.return_value = (
            mock_modified_relevancy_table  # Return modified table
        )

        relevancy_table = prepare_relevancy_table(
            mock_session_state.relevancy_fqn, mock_session
        )

        mock_session.table.assert_called_once_with("mock_relevancy_fqn")
        mock_relevancy_table.withColumn.assert_called_once_with(
            QUERY_ID, mock_session_state.md5_hash(mock_relevancy_table[QUERY])
        )

        # Expect the modified table as the result
        self.assertEqual(relevancy_table, mock_modified_relevancy_table)

    @patch("streamlit.session_state", new_callable=MagicMock)
    def test_extract_and_dedupe_goldens(self, mock_session_state):
        self.maxDiff = None
        mock_session_state.rel_scores = {}
        mock_session_state.colors = {}
        relevance_color_mapping = {
            0: "lightcoral",
            1: "lightyellow",
            2: "lightgreen",
            3: "lightgreen",
        }

        # Setup mock rows in relevancy_table with duplicate (query_id, doc_id) pairs
        mock_rows = [
            {QUERY_ID: "q1", DOC_ID: "d1", RELEVANCY: 1},
            {
                QUERY_ID: "q1",
                DOC_ID: "d1",
                RELEVANCY: 2,
            },  # Higher score, should replace
            {QUERY_ID: "q1", DOC_ID: "d2", RELEVANCY: 1},
            {QUERY_ID: "q2", DOC_ID: "d1", RELEVANCY: 3},
            {
                QUERY_ID: "q2",
                DOC_ID: "d1",
                RELEVANCY: 2,
            },  # Lower score, should be ignored
        ]

        # Run function
        mock_relevancy_table = MagicMock()
        mock_relevancy_table.collect.return_value = mock_rows
        raw_goldens = extract_and_dedupe_goldens(mock_relevancy_table)

        # Assertions on deduplicated results
        expected_raw_goldens = {
            "q1": [("d1", 2), ("d2", 1)],  # d1 has the updated score of 2
            "q2": [("d1", 3)],  # d1 has the highest score of 3
        }
        print("yey", raw_goldens, expected_raw_goldens)

        self.assertEqual(raw_goldens, expected_raw_goldens)

        # Verify session_state updates
        self.assertEqual(mock_session_state.rel_scores["q1"]["d1"], 2)
        self.assertEqual(mock_session_state.rel_scores["q2"]["d1"], 3)
        self.assertEqual(
            mock_session_state.colors["q1"]["d1"], relevance_color_mapping[2]
        )
        self.assertEqual(
            mock_session_state.colors["q2"]["d1"], relevance_color_mapping[3]
        )

    def test_prepare_golden_scores(self):
        raw_goldens = {
            "q1": [("d1", 3), ("d2", 2)],
            "q2": [("d3", 1)],
        }
        result = prepare_golden_scores(raw_goldens)

        self.assertIn("q1", result)
        self.assertEqual(result["q1"]["d1"]["rank"], 0)
        self.assertEqual(result["q1"]["d1"]["score"], 3)
        self.assertEqual(result["q2"]["d3"]["rank"], 0)
        self.assertEqual(result["q2"]["d3"]["score"], 1)

    @patch("eval.calculate_metrics")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    def test_evaluate_queries(self, mock_empty, mock_progress, mock_calculate_metrics):
        mock_query_table = MagicMock(spec=Table)
        mock_query_table.collect.return_value = [{QUERY_ID: "q1"}, {QUERY_ID: "q2"}]
        mock_scrape_df = MagicMock(spec=DataFrame)
        mock_scrape_df.collect.side_effect = [
            [{DOC_ID: "d1"}, {DOC_ID: "d2"}],  # For q1
            [{DOC_ID: "d3"}],  # For q2
        ]
        goldens = {
            "q1": {"d1": {"rank": 0, "score": 3}, "d2": {"rank": 1, "score": 2}},
            "q2": {"d3": {"rank": 0, "score": 1}},
        }
        mock_calculate_metrics.return_value = {
            HIT_RATE: 0.8,
            NDCG: 0.7,
            PRECISION: 0.6,
            RECALL: 0.5,
        }

        result = evaluate_queries(mock_query_table, mock_scrape_df, goldens)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][QUERY_ID], "q1")
        self.assertEqual(result[0][HIT_RATE], 0.8)
        self.assertEqual(result[1][QUERY_ID], "q2")
        self.assertEqual(result[1][RECALL], 0.5)


if __name__ == "__main__":
    unittest.main()
