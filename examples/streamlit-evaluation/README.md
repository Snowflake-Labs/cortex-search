# \[Cortex Search\] Eval Tool Instructions

Welcome to Cortex Search’s eval tool\! The goal of this tool is to allow you to evaluate how well your cortex search service performs on a provided query set and relevancy table (golden set). Below are instructions on how to use the tool:

1. Set up the eval tool:  
   1. Navigate to the snowflakelabs repo.  
   2. Copy and paste the eval code into your own streamlit app in your snowflake account. You can create a streamlit app by following these instructions: [https://docs.snowflake.com/en/developer-guide/streamlit/create-streamlit-ui](https://docs.snowflake.com/en/developer-guide/streamlit/create-streamlit-ui)  
   3. Ensure that you have the following packages set for the streamlit app (top left corner in the streamlit code editor):  
      1. `matplotlib` – 3.7.2  
      2. `pandas` – 2.0.3  
      3. `snowflake` – 0.13.0  
   4. Hit the “Run” button in the top right.  
2. Input the cortex search service (CSS) in fully qualified name format (\<db\>.\<schema\>.\<object\>)  
   1. Note that, in addition to the text column, the cortex search service being evaluated should also have a corresponding `ID` column uniquely identifying each text in the cortex search service.  
3. Input the query set in fully qualified name format (\<db\>.\<schema\>.\<object\>).  
   1. This is the set of queries to retrieve results from the specified cortex search service. This table should have 1 column:  
   QUERY (string) – The query to retrieve results from the cortex search service.

         | QUERY |
         | :---- |
         |       |

4. Input the CSS column representing the ID of an entry in the service.  
5. Input the CSS column representing the text of an entry in the service.  
6. The eval tool will ask whether you have a scrape or want to generate a scrape. If you are a first time user, we recommend generating a scrape (clicking “No”).  
   1. If you have a scrape, select “Yes”. The tool will then ask to provide the run id of the scrape you want to use to generate metrics. This tool assumes that the previous scrape is stored in the table \<db\>.\<schema\>.CORTEX\_SEARCH\_SCRAPE (\<db\> and \<schema\> are inferred from the fully qualified name of the cortex search service) with a corresponding run id. If the scrape you want to use is not in the table, please add it using the following steps:  
      1. Transform your scrape such that it matches the following schema. The columns are described below:  
         1. RunID (string) – a unique identifier for the scrape. The RunID can be generated with the following:

         `hashlib.md5(str(datetime.now()).encode('utf-8')).hexdigest()`

         2. QueryID (string) – the ID for the corresponding query being scraped. This is an md5 hash of the original query.  
         3. DocID (string) – the unique identifier for a doc in the corpus used to create the cortex search service.  
         4. Rank (int) – the rank of the doc returned by the cortex search service.  
         5. Debug Signals (string) – a json encoded string with all the signals associated with the result that is returned.

         | RUN\_ID | QUERY\_ID | DOC\_ID | RANK | DEBUG\_SIGNALS |
         | :---- | :---- | :---- | :---- | :---- |
         |       |       |       |       |       |


      2. Append it to the table  \<db\>.\<schema\>.CORTEX\_SEARCH\_SCRAPE  
         1. By default, \<db\>.\<schema\>.CORTEX\_SEARCH\_SCRAPE is used to store scrapes. The name can be overridden if you want to use a different table.   
   2. If you do not have a scrape, select “No”. The tool will generate a scrape and output the corresponding run id.  
      1. After selecting No, provide the result limit. The result limit is the max number of results you want retrieved per query from the search service.  
      2. Click “Generate Scrape” to generate your scrape.  
      3. The output of the scrape will be stored in \<db\>.\<schema\>.CORTEX\_SEARCH\_SCRAPE by default where the \<db\> and \<schema\> are inferred from the cortex search service’s fully qualified name. If a separate table wants to be used, the name can be overridden.   
7. Input the relevancy table in fully qualified name format in fully qualified name format (\<db\>.\<schema\>.\<object\>).  
   1. This is the golden set to compare the performance of the cortex search service with. This table should have 3 columns:  
      1. Query (string) – The query to retrieve results from the cortex search service.  
      2. DocID (string) – The doc ID of the golden result.  
         1. This can be any unique identifier for the result doc.  
      3. Relevancy (int) – A score representing how relevant the result is to the query  
         1. This value should be either 0, 1, 2, 3:  
            1. 0 \= No relevance to the query. This means that the result does not contain any relevant information to the query.  
            2. 1 \= Low relevance to the query. This means that the result contains very slightly related information to the query.  
            3. 2 \= Medium relevance to the query. This means that the result contains information that is relevant to the query, but doesn’t fully answer the query.  
            4. 3 \= High relevance to the query. This means that the result contains information that does answer the query.

         | QUERY | DOC\_ID | RELEVANCY |
         | :---- | :----   | :----     |
         |       |         |           |

8. By default, the metrics calculated will be stored in \<db\>.\<schema\>.CORTEX\_SEARCH\_METRICS, where the \<db\> and \<schema\> are inferred from the cortex search service’s fully qualified name. If a separate table wants to be used, the name can be overridden.  
9. (Optional) Add a comment to tag the metrics being calculated on the scrape. This can help identify metric calculations in the future.  
10. Click “Run Eval” for metrics to be calculated.  
11. Eval metrics will be stored in the metrics table provided in (8) with the following schema. The columns are described below:  
    1. RunID (string) – the RunID of the scrape the metrics were calculated on.  
    2. RunMetadata (string) – a json encoded string holding information about the scrape and metrics calculation itself. Information like timestamp of scrape, number of queries scraped, experimental args for the service used, the names of the inputs (cortex search service, queryset, relevancy table), run comment, etc. will be recorded here.  
    3. NDCG – the ndcg@k metrics calculated. Stored as a json object for different k values.  
    4. Precision – the precision@k metrics calculated. Stored as a json object for different k values  
    5. Recall – the recall@k metrics calculated. Stored as a json object for different k values  
    6. HitRate – the hitrate@k metrics calculated. Stored as a json object for different k values

      | RUN\_ID | RUN\_METADATA | NDCG | PRECISION | RECALL | HITRATE |
      | :---- | :---- | :---- | :---- | :---- | :---- |
      |       |       |       |       |       |       |

12. After running eval, the tool will describe and display visualizations showcasing the performance of the cortex search service on the queryset provided based on the relevancy table provided.  
    1. Both aggregate and query level metrics will be displayed for in depth analysis.  
    2. Additionally, if you want to take a closer look at other queries, input the corresponding query ID into the text box labeled “Input Query ID to learn more”. The query ID for a query can be found in the “Per Query Scores” table.