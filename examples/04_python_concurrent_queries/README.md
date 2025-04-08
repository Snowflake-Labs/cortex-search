
#  Executing concurrent Cortex Search queries with throttling

This directory contains example usage for the Cortex Search REST API
via the `snowflake.core` python package. `snowflake.core` supports
authentication and connection to a Snowflake account through several
different mechanisms, a few of which are outlined in the examples.

Notably, the Cortex Search API is only available in versions of
`snowflake.core >= 0.8.0`.

## Prerequisites

Before you can run the examples, ensure you have the following
prerequisites installed:

- Python 3.11
- pip (Python package installer)

Additionally, you must have access to a Snowflake account and the
required permissions to query the Cortex Search Service at the
specified database and schema.

## Installation

First, clone this repository to your local machine using git and navigate to this directory:

```
git clone https://github.com/snowflake-labs/cortex-search.git
cd cortex-search/examples/04_python_concurrent_queries
```

Install the necessary Python dependencies by running:

```
pip install -r requirements.txt
```

## Usage

`concurrent-example.py` is an example for running concurrent search
requests on your cortex search service along with throttling the
request rate when the server is busy.

This example collects default connection parameters from a file.  Add the
following lines to a file named `.env` in the current directory:

```
SNOWFLAKE_ACCOUNT=AID123456
SNOWFLAKE_USER=myself
SNOWFLAKE_AUTHENTICATOR=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_ROLE=my_role
SNOWFLAKE_DATABASE=my_db
SNOWFLAKE_SCHEMA=my_schema
SNOWFLAKE_CORTEX_SEARCH_SERVICE=my_service
```

Add the queries you want to run concurrently to a file named
`queries`, also in the current directory, one query per line:

```
riding shotgun
hello
what's going on
pet sounds
blue
songs in the key of life
```

You can then run `concurrent-example.py` file to print search results:

```
python concurrent-example.py --columns col1 col2
```

You can override the connection parameters specified in `.env` file with command line parameters. Try:


```
python concurrent-example.py -h
```

to enumerate the list of command line options.
