# Agentic Search with Cortex Search

This quickstart walks you through building an agentic search experience over a collection of business and logistics PDFs. You will:

1. Upload PDFs to a Snowflake internal stage
2. Parse PDFs into structured markdown using `AI_PARSE_DOCUMENT`
3. Chunk the parsed documents using `SPLIT_TEXT_MARKDOWN_HEADER`
4. Create a multi-index Cortex Search Service
5. Create an agent powered by the search service

## Prerequisites

- A Snowflake account with access to the following preview features (reach out to your account team to enable):
  - [Multi-index Cortex Search Services](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/multi-index-service)
  - [Agentic Search](https://docs.snowflake.com/en/LIMITEDACCESS/agentic-search)
- A warehouse with sufficient compute
- A database and schema to work in (replace `<your_db>` and `<your_schema>` placeholders below with your own)

## Sample Data

Download the sample PDFs (~5,000 synthetic business and logistics documents) from [Google Drive](<LINK_TODO>). The dataset includes contracts, invoices, purchase orders, shipping manifests, operational reports, and delivery confirmations.

## Step 1: Upload PDFs to a Stage

Create an internal stage and upload the sample PDFs.

```sql
CREATE OR REPLACE STAGE <your_db>.<your_schema>.agentic_search_pdfs
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  DIRECTORY = (ENABLE = TRUE);
```

Upload the PDFs from `sample_pdfs/` using [SnowSQL](https://docs.snowflake.com/en/user-guide/snowsql) or the Snowsight UI:

```bash
-- Using SnowSQL
snowsql -q "PUT file:///<path_to>/sample_pdfs/*.pdf @<your_db>.<your_schema>.agentic_search_pdfs AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
```

Verify the upload:

```sql
LIST @<your_db>.<your_schema>.agentic_search_pdfs;
```

## Step 2: Parse PDFs into a Table

Use [`AI_PARSE_DOCUMENT`](https://docs.snowflake.com/en/sql-reference/functions/ai_parse_document) with `LAYOUT` mode to extract structured markdown content from each PDF.

```sql
CREATE OR REPLACE TABLE <your_db>.<your_schema>.agentic_search_docs AS
SELECT
    RELATIVE_PATH AS file_name,
    REPLACE(SPLIT_PART(RELATIVE_PATH, '.', 1), '/', '') AS doc_id,
    PARSE_JSON(
        AI_PARSE_DOCUMENT(
            TO_FILE(@<your_db>.<your_schema>.agentic_search_pdfs, RELATIVE_PATH),
            {'mode': 'LAYOUT'}
        )
    ):content::STRING AS document_content,
    -- The columns below are placeholders; adjust based on your document metadata.
    -- If your PDFs have metadata like execution_date or status, extract them here.
    NULL::DATE AS execution_date,
    NULL::STRING AS status
FROM DIRECTORY(@<your_db>.<your_schema>.agentic_search_pdfs);
```

> **Note:** `AI_PARSE_DOCUMENT` in `LAYOUT` mode extracts text along with structural content such as tables and headers. This is recommended for business documents with complex formatting. For simpler documents, `OCR` mode (the default) may suffice.

Verify the parsed documents:

```sql
SELECT doc_id, file_name, LEFT(document_content, 500) AS preview
FROM <your_db>.<your_schema>.agentic_search_docs
LIMIT 10;
```

## Step 3: Chunk the Documents

Split the parsed markdown content into smaller chunks using [`SPLIT_TEXT_MARKDOWN_HEADER`](https://docs.snowflake.com/en/sql-reference/functions/split_text_markdown_header). This preserves document structure by keeping header context with each chunk.

```sql
CREATE OR REPLACE TABLE <your_db>.<your_schema>.agentic_search_chunks AS (
    SELECT
        docs.*,
        chunks.value['headers'] || '\n\n' || chunks.value['chunk']::TEXT AS chunk,
        CONCAT(doc_id, '.', chunks.index) AS unique_chunk_id
    FROM
        <your_db>.<your_schema>.agentic_search_docs docs,
        LATERAL FLATTEN(
            SNOWFLAKE.CORTEX.SPLIT_TEXT_MARKDOWN_HEADER(
                document_content,
                OBJECT_CONSTRUCT('#', 'header_1', '##', 'header_2', '###', 'header_3'),
                1800,
                300
            )
        ) chunks
);
```

Verify the chunks:

```sql
SELECT * FROM <your_db>.<your_schema>.agentic_search_chunks LIMIT 100;
SELECT COUNT(*) FROM <your_db>.<your_schema>.agentic_search_chunks;
```

## Step 4: Create a Multi-Index Cortex Search Service

Create the search service with both text and vector indexes. The multi-index configuration enables agentic capabilities like SQL-based filtering, aggregation, and extraction during search.

```sql
CREATE OR REPLACE CORTEX SEARCH SERVICE <your_db>.<your_schema>.agentic_search_service
    TEXT INDEXES CHUNK, DOC_ID, FILE_NAME, STATUS, UNIQUE_CHUNK_ID
    VECTOR INDEXES CHUNK (MODEL='snowflake-arctic-embed-l-v2.0'),
                   FILE_NAME (MODEL='snowflake-arctic-embed-l-v2.0')
    ATTRIBUTES EXECUTION_DATE, STATUS, DOC_ID, FILE_NAME, UNIQUE_CHUNK_ID
    WAREHOUSE = '<your_warehouse>'
    TARGET_LAG = '1 day'
    AS (
        SELECT
            CHUNK,
            DOC_ID,
            FILE_NAME,
            DOCUMENT_CONTENT,
            EXECUTION_DATE,
            STATUS,
            UNIQUE_CHUNK_ID
        FROM <your_db>.<your_schema>.agentic_search_chunks
    );
```

> **Note:** `DOCUMENT_CONTENT` is included in the source query so it is available in the base table for downstream context, but it is not indexed by the search service.

## Step 5: Create an Agent

Create an agent that uses the multi-index search service. The agent specification defines the orchestration model, tool configuration, and column metadata that the agent uses to understand and query your data.

```sql
CREATE OR REPLACE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.<your_agent_name> FROM SPECIFICATION $$
{
    "models": {
        "orchestration": "claude-sonnet-4-5"
    },
    "instructions": {
        "orchestration": "You are a helpful assistant that answers questions about business and logistics documents including purchase orders, invoices, contracts, shipping manifests, operational reports, and delivery confirmations. Use the search tool to find relevant information before answering."
    },
    "tools": [
        {
            "tool_spec": {
                "type": "cortex_search",
                "name": "business_eval_multi_index",
                "description": "Search service containing business and logistics documents chunked into smaller documents. At a high level, each document (CHUNK column) is a chunk of text from the business document (purchase orders, invoices, contracts, shipping manifests, operational reports, delivery confirmations). The DOCUMENT_CONTENT column is the original full document content (IMPORTANT: DOCUMENT_CONTENT is not indexed in the search service, it is only in the base table). Therefore, the CHUNK text can be found in the DOCUMENT_CONTENT column. IMPORTANT: DOCUMENT_CONTENT column should be used for context of the entire document AFTER search. IMPORTANT: always use use_sql_for_search."
            }
        }
    ],
    "tool_resources": {
        "business_eval_multi_index": {
            "name": "<your_db>.<your_schema>.agentic_search_service",
            "search_service": "<your_db>.<your_schema>.agentic_search_service",
            "database_schema": "<your_db>.<your_schema>",
            "is_multi_index": true,
            "columns_and_descriptions": {
                "CHUNK": {
                    "description": "The chunk of text from a business/logistics document. There can be multiple chunks from the same document because documents are long. In the base table, there is a DOCUMENT_CONTENT column where this chunk originally came from. IMPORTANT: The DOCUMENT_CONTENT column is not indexed in the search service and cannot be searched over or filtered on.",
                    "type": "string",
                    "searchable": true,
                    "filterable": false
                },
                "DOC_ID": {
                    "description": "Business document identifier. IMPORTANT: This is not unique to each chunk. Each chunk in this index is a chunk of text from the document that has this doc id. This corresponds to the DOCUMENT_CONTENT column which is not indexed in the search service.",
                    "type": "string",
                    "searchable": true,
                    "filterable": true
                },
                "FILE_NAME": {
                    "description": "The file name of the original business document.",
                    "type": "string",
                    "searchable": true,
                    "filterable": false
                },
                "EXECUTION_DATE": {
                    "description": "The execution date of the business document (filterable). This is a date field that can be filtered using @gte and @lte operators. IMPORTANT: any time date is being discussed, this column should be used for filtering.",
                    "type": "date",
                    "searchable": false,
                    "filterable": true
                },
                "STATUS": {
                    "description": "The status of the business document (e.g., Active, Completed, Pending, Cancelled, On Hold). This is searchable and filterable.",
                    "type": "string",
                    "searchable": true,
                    "filterable": true
                },
                "UNIQUE_CHUNK_ID": {
                    "description": "The unique chunk id of the business document. This is unique to each chunk. The format is {DOC_ID}.{CHUNK_ID}.",
                    "type": "string",
                    "searchable": true,
                    "filterable": true
                },
                "DOCUMENT_CONTENT": {
                    "description": "The full text of the business document. The value in CHUNK is found in DOCUMENT_CONTENT. The DOC_ID column is unique to each DOCUMENT_CONTENT value. IMPORTANT: This is not indexed in the search service, it is only in the base table, and should only be used after search for context of the entire document.",
                    "type": "string",
                    "searchable": false,
                    "filterable": false
                }
            },
            "max_results": 800,
            "execution_environment": {
                "type": "warehouse",
                "warehouse": "<your_warehouse>"
            },
            "id_column": "UNIQUE_CHUNK_ID",
            "base_table": "<your_db>.<your_schema>.agentic_search_chunks",
            "base_table_columns": [
                "CHUNK",
                "DOC_ID",
                "FILE_NAME",
                "DOCUMENT_CONTENT",
                "EXECUTION_DATE",
                "STATUS",
                "UNIQUE_CHUNK_ID"
            ],
            "doc_id_column": "DOC_ID",
            "doc_to_chunk_fanout": 4
        }
    }
}
$$;
```

## Using the Agent

Once created, you can interact with the agent through Snowsight:

1. Navigate to **AI & ML > Snowflake Intelligence**
2. Select your agent from the agent dropdown
3. Choose **Agentic Search** as the source
4. Ask questions about your business documents

Example queries:
- "What are the active contracts?"
- "Show me all invoices from Q1 2025"
- "Find shipping manifests with status Pending"
- "Summarize the operational reports for last month"
