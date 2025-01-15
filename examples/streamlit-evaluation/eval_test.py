import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock
import math
from eval import (
    hit_rate,
    sdcg,
    precision,
    _dcg,
    generate_and_store_scrape,
    prepare_query_df,
    perform_scrape_for_eval,
    perform_scrape_for_autotune,
    store_scrape_results,
    validate_scrape,
    get_result_limit,
    prepare_relevancy_df,
    extract_and_dedupe_goldens,
    prepare_golden_scores,
    evaluate_queries,
    compute_fusion_score_from_service,
    QUERY_ID,
    QUERY,
    HIT_RATE,
    RELEVANCY,
    RUN_ID,
    DOC_ID,
    SDCG,
    PRECISION,
    DEBUG_PER_RESULT,
    RESPONSE_RESULTS,
    DEBUG_SIGNALS,
    DEBUG,
    SLOWMODE,
    EmbeddingMultiplier,
    RerankingMultiplier,
    TopicalityMultiplier,
    RERANK_WEIGHTS,
    DEFAULT_EMBEDDING_MULTIPLIER,
)
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.snowpark import Table, DataFrame
import hashlib


TEXT = "text"


class TestMetrics(TestCase):
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

    def test_sdcg(self):
        # Calculate DCG and SDCG values for given results and golden data
        expected_sdcg = _dcg(self.results, self.golden_to_score) / (
            3.0 * sum(1.0 / math.log2(i + 2.0) for i in range(4))
        )
        self.assertAlmostEqual(
            sdcg(3.0, self.results, self.golden_to_score), expected_sdcg, places=5
        )

        # Test SDCG with empty results
        self.assertEqual(sdcg(3.0, self.empty_results, self.golden_to_score), 0.0)

        # Test SDCG with empty golden set
        self.assertEqual(sdcg(3.0, self.results, self.empty_golden), 0.0)

    def test_precision(self):
        # 2 relevant documents (doc1, doc2), out of 4 total -> precision = 2/4 = 0.5
        self.assertAlmostEqual(precision(self.results, self.golden_to_score), 0.5)

        # No relevant documents, precision should be 0
        self.assertEqual(precision(["doc4"], self.golden_to_score), 0.0)

        # Test precision with empty results
        self.assertEqual(precision(self.empty_results, self.golden_to_score), 0.0)

    def test_dcg(self):
        # Test DCG for correct calculation
        expected_dcg = 3 / math.log2(2) + 2 / math.log2(3)  # score of doc1 and doc2
        self.assertAlmostEqual(
            _dcg(self.results, self.golden_to_score), expected_dcg, places=5
        )

        # Test DCG with empty results
        self.assertEqual(_dcg(self.empty_results, self.golden_to_score), 0.0)


class TestScrapeFlow(TestCase):
    @patch("eval.get_active_session")
    @patch("eval.Root")
    @patch("streamlit.session_state", new_callable=MagicMock)
    @patch("eval.prepare_query_df")
    @patch("eval.perform_scrape_for_eval")
    @patch("eval.store_scrape_results")
    def test_generate_and_store_scrape(
        self,
        mock_store_scrape_results,
        mock_perform_scrape_for_eval,
        mock_prepare_query_df,
        mock_session_state,
        mock_root,
        mock_get_active_session,
    ):
        # Mock session state properties
        mock_session_state.start_time = datetime(2024, 1, 1)

        # Mock return values
        mock_query_table = MagicMock(name="query_table")
        mock_prepare_query_df.return_value = mock_query_table
        mock_scrape_out = [{"example_key": "example_value"}]
        mock_perform_scrape_for_eval.return_value = mock_scrape_out

        # Mock session and root
        mock_session = mock_get_active_session.return_value
        mock_root_instance = mock_root.return_value

        # Call function
        generate_and_store_scrape(mock_session, mock_root_instance)

        # Assertions
        mock_prepare_query_df.assert_called_once()
        mock_perform_scrape_for_eval.assert_called_once_with(
            mock_session, mock_query_table, mock_root_instance
        )
        mock_store_scrape_results.assert_called_once_with(mock_scrape_out)

    @patch("eval.get_active_session")  # Mock get_active_session
    @patch("eval.get_session")  # Mock get_session
    @patch("eval.Root")  # Mock Root
    @patch("streamlit.session_state", new_callable=MagicMock)  # Mock st.session_state
    def test_prepare_query_df_creates_query_id_if_missing(
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
        mock_query_table.with_column.return_value = mock_query_table

        # Setup the session.table mock to return mock_query_table
        mock_session.table.return_value = mock_query_table

        # Call the function
        result = prepare_query_df(mock_session)  # Pass the mock session

        # Assertions
        mock_session.table.assert_called_once_with(
            "mock_fqn"
        )  # Check if session.table was called with the correct fqn
        mock_query_table.with_column.assert_called_once_with(
            QUERY_ID, mock_session_state.md5_hash(mock_query_table[QUERY])
        )  # Ensure with_column was called to add QUERY_ID using md5_hash
        self.assertEqual(
            result, mock_query_table
        )  # Ensure the result is the mock query_table

    @patch(
        "streamlit.session_state", new_callable=MagicMock
    )  # Mock streamlit session_state
    @patch("eval.Session", new_callable=MagicMock)  # Mock eval.Session
    @patch("eval.perform_scrape", new_callable=MagicMock)  # Mock perform_scrape
    def test_perform_scrape_for_eval(
        self, mock_perform_scrape, mock_session, mock_session_state
    ):
        # Mock session state properties
        mock_session_state.css_id_given = True  # Simulate that CSS ID is given
        mock_session_state.css_id_col = DOC_ID
        mock_session_state.css_text_col = "text"  # Adjust based on your code
        mock_session_state.css_fqn = "mock_db.mock_schema.mock_service"

        # Mock input query DataFrame
        mock_query_df = MagicMock()
        mock_query_df.collect.return_value = [{QUERY: "test query", "QUERY_ID": "1"}]

        # Mock perform_scrape output
        mock_perform_scrape.return_value = {
            "1": {
                QUERY: "test query",
                RUN_ID: "mock_run_id",
                RESPONSE_RESULTS: [
                    {
                        mock_session_state.css_text_col: "sample text",
                        DEBUG_PER_RESULT: {"score": 0.9},
                    }
                ],
            }
        }

        # Expected output
        expected_output = [
            {
                QUERY: "test query",
                RUN_ID: "mock_run_id",
                QUERY_ID: "1",
                DOC_ID: hashlib.md5("sample text".encode("utf-8")).hexdigest(),
                "RANK": 1,
                DEBUG_SIGNALS: {"score": 0.9},
                "TEXT": "sample text",
            }
        ]

        # Mock the DataFrame creation
        mock_scrape_df = MagicMock()
        mock_session.create_dataframe.return_value = mock_scrape_df

        # Call the function under test
        result_df = perform_scrape_for_eval(
            mock_session, mock_query_df, MagicMock(), run_id="mock_run_id"
        )

        mock_session.create_dataframe.assert_called_once_with(
            expected_output
        )  # Check DataFrame creation

        # Ensure returned DataFrame matches the mocked DataFrame
        self.assertEqual(result_df, mock_scrape_df)

    @patch("streamlit.session_state", new_callable=MagicMock)
    def test_store_scrape_results(self, mock_session_state):
        # Mock session state properties
        mock_session_state.scrape_fqn = "mock_db.mock_schema.mock_table"
        mock_session_state.start_time = datetime(2024, 1, 1, 12, 0, 0)

        # Mock the scrape DataFrame
        mock_scrape_df = MagicMock(name="scrape_df")

        # Mock datetime to control timing
        mock_now = datetime(2024, 1, 1, 12, 0, 10)
        with patch("eval.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now

            # Call the function under test
            store_scrape_results(mock_scrape_df)

            # Assertions for the DataFrame write operation
            mock_scrape_df.write.mode.assert_called_once_with("append")
            mock_scrape_df.write.mode().save_as_table.assert_called_once_with(
                "mock_db.mock_schema.mock_table"
            )

            # Assertions for session state and success message
            duration = mock_now - mock_session_state.start_time
            assert duration.total_seconds() == 10.0


class TestPrepareRelevancyTable(TestCase):
    @patch("streamlit.session_state", new_callable=MagicMock)
    @patch("eval.Session")
    def test_prepare_relevancy_df(self, mock_session, mock_session_state):
        # Mock session state properties
        mock_session_state.md5_hash = MagicMock(side_effect=lambda x: f"hash_{x}")
        mock_session_state.css_text_col = TEXT

        # Mock relevancy table with QUERY_ID and DOC_ID present
        mock_table_with_ids = MagicMock()
        mock_table_with_ids.columns = [QUERY_ID, DOC_ID]
        mock_table_with_ids.withColumn.return_value = (
            mock_table_with_ids  # Mock chainable behavior
        )

        mock_session.table.return_value = mock_table_with_ids

        # Call the function under test
        result_table = prepare_relevancy_df("relevancy_fqn", mock_session)

        # Assert QUERY_ID and DOC_ID were cast to string
        mock_table_with_ids.withColumn.assert_any_call(
            QUERY_ID, mock_table_with_ids[QUERY_ID].cast("string")
        )
        mock_table_with_ids.withColumn.assert_any_call(
            DOC_ID, mock_table_with_ids[DOC_ID].cast("string")
        )

        # Assert no new columns were added
        self.assertEqual(result_table, mock_table_with_ids)
        mock_session_state.md5_hash.assert_not_called()

    @patch("streamlit.session_state", new_callable=MagicMock)
    @patch("eval.Session")
    def test_prepare_relevancy_df_missing_columns(
        self, mock_session, mock_session_state
    ):
        # Mock session state properties
        mock_session_state.md5_hash = MagicMock(side_effect=lambda x: f"hash_{x}")
        mock_session_state.css_text_col = TEXT

        # Mock relevancy table missing QUERY_ID and DOC_ID
        mock_table_missing_ids = MagicMock()
        mock_table_missing_ids.columns = [QUERY]
        mock_table_missing_ids.withColumn.return_value = (
            mock_table_missing_ids  # Mock chainable behavior
        )

        mock_session.table.return_value = mock_table_missing_ids

        # Call the function under test
        result_table = prepare_relevancy_df("relevancy_fqn", mock_session)

        # Assert QUERY_ID was added using md5_hash on QUERY
        mock_table_missing_ids.withColumn.assert_any_call(
            QUERY_ID, mock_session_state.md5_hash(mock_table_missing_ids[QUERY])
        )

        # Assert DOC_ID was added using md5_hash on css_text_col
        mock_table_missing_ids.withColumn.assert_any_call(
            DOC_ID,
            mock_session_state.md5_hash(
                mock_table_missing_ids[mock_session_state.css_text_col]
            ),
        )

        # Assert result_table matches the transformed table
        self.assertEqual(result_table, mock_table_missing_ids)


class TestEvalFlow(TestCase):
    @patch("streamlit.session_state", new_callable=MagicMock)
    @patch("eval.Session")
    def test_prepare_query_df(self, mock_session, mock_session_state):
        # Mock session and input table
        mock_query_table = MagicMock()
        mock_session.table.return_value = mock_query_table
        mock_query_table.columns = ["query_column"]  # QUERY_ID missing initially

        # Mock behavior of session_state's md5_hash
        mock_session_state.queryset_fqn = "mock_queryset_fqn"
        mock_session_state.md5_hash.return_value = "mock_hash"

        # Mock the with_column behavior to add QUERY_ID
        mock_query_table.with_column.return_value = mock_query_table

        # Call prepare_query_df
        query_df = prepare_query_df(mock_session)

        # Verify the correct calls
        mock_session.table.assert_called_once_with("mock_queryset_fqn")
        mock_query_table.with_column.assert_called_once_with(
            "QUERY_ID", mock_session_state.md5_hash(mock_query_table["query_column"])
        )
        self.assertEqual(query_df, mock_query_table)

    @patch("streamlit.session_state", new_callable=MagicMock)
    @patch("eval.Session")
    def test_prepare_query_df_with_query_id(self, mock_session, mock_session_state):
        # Mock session and input table
        mock_query_table = MagicMock()
        mock_session.table.return_value = mock_query_table
        mock_query_table.columns = [
            "query_column",
            "QUERY_ID",
        ]  # QUERY_ID already exists

        # Mock behavior of session_state's md5_hash
        mock_session_state.queryset_fqn = "mock_queryset_fqn"

        # Call prepare_query_df
        query_df = prepare_query_df(mock_session)

        # Verify the correct calls
        mock_session.table.assert_called_once_with("mock_queryset_fqn")
        mock_query_table.with_column.assert_not_called()  # QUERY_ID already exists
        self.assertEqual(query_df, mock_query_table)

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

    @patch("streamlit.session_state", new_callable=MagicMock)
    def test_prepare_golden_scores(self, mock_session_state):
        mock_session_state.relevancy_provided = True
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
    @patch("streamlit.session_state", new_callable=MagicMock)
    def test_evaluate_queries(
        self, mock_session_state, mock_empty, mock_progress, mock_calculate_metrics
    ):
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
            SDCG: 0.7,
            PRECISION: 0.6,
        }
        mock_session_state.idcg_factor = 3.0

        result = evaluate_queries(mock_query_table, mock_scrape_df, goldens)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][QUERY_ID], "q1")
        self.assertEqual(result[0][HIT_RATE], 0.8)
        self.assertEqual(result[1][QUERY_ID], "q2")


class TestAutotuneFlow(TestCase):
    @patch("streamlit.session_state", new_callable=MagicMock)  # Mock st.session_state
    @patch("eval.generate_docid")
    @patch("eval.perform_scrape")
    @patch("eval.Root")
    def test_perform_scrape_for_autotune(
        self,
        mock_root,
        mock_perform_scrape,
        mock_generate_docid,
        mock_session_state,
    ):
        # Mock parameters and results
        mock_experimental_params = {
            DEBUG: True,
            SLOWMODE: True,
            RERANK_WEIGHTS: {
                RerankingMultiplier: 1.4,
                EmbeddingMultiplier: DEFAULT_EMBEDDING_MULTIPLIER,
                TopicalityMultiplier: 1.0,
            },
        }

        # Mock return values
        mock_query_df = MagicMock(spec=DataFrame, name="query_df")
        mock_scrape_out = {
            "123": {
                QUERY: "abc",
                RUN_ID: "xyz",
                RESPONSE_RESULTS: [{"TEXT": "text"}],
            }
        }
        mock_perform_scrape.return_value = mock_scrape_out
        mock_generate_docid.return_value = "mnp"

        # Mock session and root
        mock_session_state.css_text_col = "TEXT"
        mock_root_instance = mock_root.return_value

        # Expected output
        expected_output = {"123": ["mnp"]}

        # Call function
        result = perform_scrape_for_autotune(
            mock_query_df, mock_root_instance, mock_experimental_params
        )

        # Assertions
        mock_perform_scrape.assert_called_once_with(
            mock_query_df,
            mock_root_instance,
            autotune=True,
            experimental_params=mock_experimental_params,
            run_id="",
        )
        mock_generate_docid.assert_called_once()
        self.assertEqual(result, expected_output)

    @patch("streamlit.session_state", new_callable=MagicMock)  # Mock st.session_state
    @patch("eval.perform_scrape_for_autotune")
    @patch("eval.sdcg")
    @patch("eval.Root")
    def test_compute_fusion_score_from_service(
        self,
        mock_root,
        mock_sdcg,
        mock_perform_scrape_for_autotune,
        mock_session_state,
    ):
        # Mock parameters and results
        mock_params = {
            RerankingMultiplier: 1.4,
            EmbeddingMultiplier: DEFAULT_EMBEDDING_MULTIPLIER,
            TopicalityMultiplier: 1.0,
        }
        mock_experimental_params = {
            DEBUG: True,
            SLOWMODE: True,
            RERANK_WEIGHTS: mock_params,
        }
        mock_doc_list = ["mnp"]
        mock_golden_set = {
            "abc": {"score": 3},
            "mnp": {"score": 8},
        }
        mock_query_to_doc_list = {"123": mock_doc_list}
        mock_goldens = {"123": mock_golden_set}

        # Mock return values
        mock_query_df = MagicMock(spec=DataFrame, name="query_df")
        mock_query_df.collect.return_value = [{QUERY_ID: "123"}]
        mock_perform_scrape_for_autotune.return_value = mock_query_to_doc_list
        mock_sdcg.return_value = 0.7

        # Mock session and root
        mock_session_state.idcg_factor = 3.0
        mock_root_instance = mock_root.return_value

        # Expected output
        expected_output = 0.7

        # Call function
        result = compute_fusion_score_from_service(
            mock_root_instance, mock_query_df, mock_goldens, mock_params
        )

        # Assertions
        mock_perform_scrape_for_autotune.assert_called_once_with(
            mock_query_df,
            mock_root_instance,
            experimental_params=mock_experimental_params,
        )
        mock_sdcg.assert_called_once_with(
            idcg_factor=3.0,
            results=["mnp"],
            golden_to_score={"abc": {"score": 3}, "mnp": {"score": 8}},
        )
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
