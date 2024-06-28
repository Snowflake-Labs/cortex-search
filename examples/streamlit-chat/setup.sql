-- ***************************************
-- *           Section 1: Setup          *
-- ***************************************

-- Create the database, schema, stage, and warehouse for the demo.
CREATE DATABASE demo_cortex_search;
CREATE SCHEMA fomc;
CREATE STAGE minutes 
	DIRECTORY = ( ENABLE = true ) 
	ENCRYPTION = ( TYPE = 'SNOWFLAKE_SSE' );
CREATE WAREHOUSE demo_cortex_search_wh;

-- Upload pdfs to demo_cortex_search.fomc.minutes from https://drive.google.com/drive/folders/1_erdfr7ZR49Ub2Sw-oGMJLs3KvJSap0d?usp=sharing


-- ***************************************
-- *      Section 2: UDTF Creation       *
-- ***************************************

-- This section creates a user-defined table function (UDTF) to parse PDFs and chunk the extracted text.
CREATE OR REPLACE FUNCTION pypdf_extract_and_chunk(file_url VARCHAR, chunk_size INTEGER, overlap INTEGER)
RETURNS TABLE (chunk VARCHAR)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.9'
HANDLER = 'pdf_text_chunker'
PACKAGES = ('snowflake-snowpark-python','PyPDF2', 'langchain')
AS
$$
from snowflake.snowpark.types import StringType, StructField, StructType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from snowflake.snowpark.files import SnowflakeFile
import PyPDF2, io
import logging
import pandas as pd

class pdf_text_chunker:

    def read_pdf(self, file_url: str) -> str:
    
        logger = logging.getLogger("udf_logger")
        logger.info(f"Opening file {file_url}")
    
        with SnowflakeFile.open(file_url, 'rb') as f:
            buffer = io.BytesIO(f.readall())
            
        reader = PyPDF2.PdfReader(buffer)   
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text().replace('\n', ' ').replace('\0', ' ')
            except:
                text = "Unable to Extract"
                logger.warn(f"Unable to extract from file {file_url}, page {page}")
        
        return text


    def process(self,file_url: str, chunk_size: int, chunk_overlap: int):

        text = self.read_pdf(file_url)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = len
        )
    
        chunks = text_splitter.split_text(text)
        df = pd.DataFrame(chunks, columns=['CHUNK'])
        
        yield from df.itertuples(index=False, name=None)
$$;

-- ***************************************
-- *      Section 3: Parse PDFs with UDTF     *
-- ***************************************

-- This section parses the PDFs, chunks the output, and inserts the chunked documents into a table.
CREATE OR REPLACE TABLE parsed_doc_chunks ( 
    relative_path VARCHAR, -- Relative path to the PDF file
    chunk VARCHAR
) AS (
    SELECT
        relative_path,
        chunks.chunk as chunk
    FROM
        directory(@DEMO_CORTEX_SEARCH.FOMC.MINUTES)
        , TABLE(pypdf_extract_and_chunk(
            build_scoped_file_url(@MINUTES, relative_path),
            2000,
            500
        )) as chunks
);


-- ***************************************
-- *    Section 4: Create Cortex Search Service    *
-- ***************************************

-- This section creates a Cortex search service with the parsed pdf data.

SELECT * FROM parsed_doc_chunks; -- preview data

CREATE OR REPLACE CORTEX SEARCH SERVICE fomc_minutes_search_service
ON minutes
ATTRIBUTES relative_path
WAREHOUSE = demo_cortex_search_wh
TARGET_LAG = '1 hour'
AS (
    SELECT
        LEFT(RIGHT(relative_path, 12), 8) as meeting_date,
        CONCAT('Meeting date: ', meeting_date, ' \nMinutes: ', chunk) as minutes,
        relative_path
    FROM parsed_doc_chunks
);

-- grant usage to public role
GRANT USAGE ON CORTEX SEARCH SERVICE fomc_minutes_search_service TO ROLE public;
GRANT USAGE ON DATABASE demo_cortex_search to role public;
GRANT USAGE ON SCHEMA demo_cortex_search.fomc to role public;
GRANT READ ON STAGE demo_cortex_search.fomc.minutes to role public;