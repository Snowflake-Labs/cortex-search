# Cortex Search in Streamlit
This guide shows you how to build a basic AI-powered search and summarization in streamlit using Snowflake Cortex.
The app allows the user to select an existing [Cortex Search Service](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/cortex-search-overview) as the backend for their semantic search experience. Optionally, the user can toggle on the `Summarize`
feature to provide an AI-generated summary of search results.

## Prerequisites
- Ensure you have [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) installed prior to running this demo

## Setup:

First, clone this repository to your local machine using git and navigate to this directory:

```
git clone https://github.com/snowflake-labs/cortex-search.git
cd cortex-search/examples/05_streamlit_ai_search_app
```


Then, create and activate the conda environment:

```
conda env create -n cortex-search -f conda_env.yml
conda activate cortex-search
``` 

Next, create a file called `.streamlit/secrets.toml` file to store aliased Snowflake connections and all the parameters needed to connect to Snowflake. See the snowflake connector docs: https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#connecting-using-the-connections-toml-file. An example is found under `.streamlit/examplke_secrets.toml`. 

## Usage:
To run locally, run:
```
streamlit run cortex_search.py
```