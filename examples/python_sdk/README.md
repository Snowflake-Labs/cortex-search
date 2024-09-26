# Cortex Search SDK usage via the `snowflake.core` library

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
cd cortex-search/examples/python_sdk
```

Install the necessary Python dependencies by running:

```
pip install -r requirements.txt
```

## Usage

### 1. Passing connection parameters explicitly or via environment variables

`simple.py` collects connection parameters from your shell
environment. These can be set like so (note: we recommend setting the
`SNOWFLAKE_ACCOUNT` using the account locator, rather than the
`org-account` format):

```
export SNOWFLAKE_ACCOUNT=AID123456
export SNOWFLAKE_USER=myself
export SNOWFLAKE_PASSWORD=pass123...
export SNOWFLAKE_ROLE=my_role
export SNOWFLAKE_DATABASE=my_db
export SNOWFLAKE_SCHEMA=my_schema
export SNOWFLAKE_CORTEX_SEARCH_SERVICE=my_service
```

However, you may also simply replace each `os.environ[".."]` with hardcoded values, if you wish.

Then, after modifying the search parameters to your liking, run the example to generate results:

```
python simple.py
```

### 2. With a connections.toml file

To set up a `connections.toml` file to store aliased Snowflake
connections and all the parameters needed to connect, see the
[snowflake connector docs](
https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#connecting-using-the-connections-toml-file)

An example is found under `example_connections.toml`, but make sure to
name yours `connections.toml` and ensure it is located at a valid
system-dependent path as specified in the Snowflake connector docs.

Then, export your Cortex Search Service name (assuming the database
and schema are already in your connection parameters):

```
export SNOWFLAKE_CORTEX_SEARCH_SERVICE=my_service
```

You can then run the `using_connections_config.py` file to print search results:

```
python using_connections_config.py
```

### 3. Executing concurrent queries with throttling

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
