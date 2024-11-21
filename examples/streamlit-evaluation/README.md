# Tutorial: Evaluating Cortex Search Quality with Streamlit

_Last updated: Nov 21, 2024_

This tutorial walks you through Cortex Search Quality evaluation in a Streamlit-in-Snowflake app. By the end of this tutorial, you will have generated and run a quantitative evaluation of search quality on your use-case for a given Cortex Search Service.

## Prerequisites

There are three objects you’ll need before beginning the evaluation process:

* A [Cortex Search Service](https://docs.snowflake.com/user-guide/snowflake-cortex/cortex-search/cortex-search-overview)  
* A query table ([example](https://docs.google.com/spreadsheets/d/1q4RMplovT5lyt-zC4Y-ncf_sl4f8qEn6ydIVwkCSqP8/edit?gid=214438211#gid=214438211), description below)  
* A relevancy table ([example](https://docs.google.com/spreadsheets/d/1q4RMplovT5lyt-zC4Y-ncf_sl4f8qEn6ydIVwkCSqP8/edit?gid=0#gid=0), description below)

### Building a query table

First, you’ll need to have a query table. Queries are the basic “inputs” in a search system. A query table has the set of questions, representative of your production workload, that you would like to retrieve answers for using Cortex Search Service. The table consists of just one column “QUERY”, and looks like the following:

| QUERY |
| :---- |
|  |

See [here](https://docs.google.com/spreadsheets/d/1q4RMplovT5lyt-zC4Y-ncf_sl4f8qEn6ydIVwkCSqP8/edit?gid=214438211#gid=214438211) for an example of a small QUERY table.

The best source of a query table is real-life customer queries. However, if you don’t have a list of real customer queries, you could consider generating them manually by reading through your document corpus and generating queries that should hit random documents, or synthetically with an LLM. One way to generate synthetic queries is to query an LLM for a given document using a prompt “*Given this text: {insert text here}, generate 5 natural search queries users might input to find this.*”

### Building a relevancy table

Then, you’ll need a relevancy table. You can think of this table as the labeled “outputs” for your query “inputs”, generated above. The relevancy table contains a set of (query, doc, relevancy) pairs, where relevancy is the perceived relevance of the document to the given query. Relevancy is represented by a *relevance score*, defined on a scale from **0 to 3**, where:

* **0 (Irrelevant):** The text does not address or relate to the query in any meaningful way.  
* **1 (Slightly Relevant):** The text contains minor or tangential information that may only loosely relate to the query.  
* **2 (Somewhat Relevant):** The text provides partial or incomplete information related to the query but does not fully satisfy the intent.  
* **3 (Perfectly Relevant):** The text fully and comprehensively addresses the query, answering it effectively and directly.

The relevancy table should have three columns, query, the text column name for the Cortex Search Service and the relevancy score of the text with respect to the query. The relevancy table should look like the following:

| QUERY | *`<CSS TEXT COLUMN NAME>`* | RELEVANCY |
| :---- | :---- | :---- |
|  |  |  |

Note: *`<CSS TEXT COLUMN NAME>`* is the name of the search column in your Cortex Search Service.

See [here](https://docs.google.com/spreadsheets/d/1q4RMplovT5lyt-zC4Y-ncf_sl4f8qEn6ydIVwkCSqP8/edit?gid=0#gid=0) for an example of a small RELEVANCY table.

Now that you have these three objects, you’re ready to get started with the evaluation process.

## Step 1: Create the Evaluation Streamlit-in-Snowflake App

* Create a new Streamlit-in-Snowflake App. For help on creating a Streamlit in Snowflake application, see [here](https://docs.snowflake.com/en/developer-guide/streamlit/create-streamlit-ui).  
* Install the following packages in the Streamlit app:  
  * `matplotlib` \>= 3.7.2  
  * `pandas` \>= 2.0.3  
  * `snowflake` \>= 0.13.0  
* Copy and paste the contents of the [eval.py file here](https://github.com/Snowflake-Labs/cortex-search/blob/main/examples/streamlit-evaluation/eval.py) into your new application and click `Run`.

## Step 2: Run the app

Now, you’re ready to get started with the evaluation process. There are in-app instructions for you to follow for running the evaluation.

## Step 3: Interpret metrics

🎉 **Evaluation Completed\!**  
You have successfully run an evaluation of the Cortex Search Service against your dataset. Here’s a summary to help you interpret the results:

1. **Section: Static results:** Key metrics summarizing the retrieval performance of the Cortex Search Service, averaged across all queries in the dataset, along with run metadata.  
2. **Section: Interactive Results (at a chosen k)**. This section computes metrics based on the chosen result limit, or `k` ,value. Choose a value of k that suits your downstream use case (commonly 5 or 10).  
   1. Aggregate Summary: High-level retrieval metrics calculated at k, with short explanations.  
   2. Per Query Metrics: Allows you to drill down into individual query performance for failed  
      1. Sort results: Click on column headers to sort. For example, if you are curious to know what queries had low NDCG, you could click on the NDCG@k column to sort such queries.  
      2. Inspect Specific Queries: Click the index of any query row to explore details. View all top-k retrieved documents for the selected query, along with their Cortex Search Service scores.  
      3. Download all queries and scores: Export all top-k results, including scores for all queries, by clicking the Download button.

## Step 4 (optional): Modify data, re-run

If you update your Cortex Search Service definition or query/relevancy tables, you can then rerun the evaluation to check the updated results.
