# Cortex Search in Streamlit

## Prerequisites

- Anaconda

## Setup:

First, clone this repository to your local machine using git and navigate to this directory:

```
git clone https://github.com/snowflake-labs/cortex-search.git
cd cortex-search/examples/streamlit-ai-search
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