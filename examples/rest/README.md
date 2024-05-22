# Cortex Search REST usage

This directory contains example usage for the Cortex Search REST API using pure python (and libraries such as `requests`). Authentication for this method requires a JWT as described below.

## Prerequisites

Before you can run the examples, ensure you have the following prerequisites installed:

- Python 3.x
- pip (Python package installer)
  Additionally, you must have access to a Snowflake account and the required permissions to query the Cortex Search Service at the specified database and schema.

## Installation

First, clone this repository to your local machine using git and navigate to this directory:

```
git clone https://github.com/snowflake-labs/cortex-search.git
cd cortex-search/examples/rest
```

Install the necessary Python dependencies by running:

```
pip install -r requirements.txt
```

## Key-pair auth configuration

Additionally, you must generate a private key for JWT auth with Snowflake as described in [this document](https://docs.snowflake.com/user-guide/key-pair-auth#configuring-key-pair-authentication).

**Note**: take note of the path to your generated RSA private key, e.g., `/path/to/my/rsa_key.p8` -- you will need to supply this as the `--private-key-path` parameter to query the service later from the command line, or list the path to the file from within a notebook.

## Usage

### 1. Notebook usage

The [examples/notebook_query.ipynb file](https://github.com/Snowflake-Labs/cortex-search/blob/main/examples/notebook_query.ipynb) shows an example of querying the service from within a Jupyter Notebook.

### 2. Python script

The `simple_query.py` example script can be executed from the command line. For instance:

```
python3 examples/simple_query.py -u https://my_org-my_account.us-west-2.aws.snowflakecomputing.com -s MY_DB.MY_SCHEMA.MY_SERVICE_NAME -q "the sky is blue" -c "description,text" -l 10 -a my_account -k /path/to/my/rsa_key.p8 -n my_name
```

**Arguments:**

- `-u`, `--url`: URL of the Snowflake instance. See [this guide](https://docs.snowflake.com/en/user-guide/admin-account-identifier#finding-the-organization-and-account-name-for-an-account) for finding your Account URL
- `-s`, `--qualified-service-name`: The fully-qualified Cortex Search Service name, in the format DATABASE.SCHEMA.SERVICE
- `-q`, `--query`: The search query string
- `-c`, `--columns`: Comma-separated list of columns to return in the results
- `-l`, `--limit`: The max number of results to return
- `-a`, `--account`: Snowflake account name. See [this guide](https://docs.snowflake.com/en/user-guide/admin-account-identifier#finding-the-organization-and-account-name-for-an-account) for finding your Account name
- `-k`, `--private-key-path`: Path to the RSA private key file for authentication.
- `-n`, `--user-name`: Username for the Snowflake account
- `-r`, `--role`: Role to use for the query. If provided, a session token scoped to this role will be created and used for authentication to the API.

The `interactive_query.py` example provides an interactive CLI that demonstrates caching the JWT used for authentication between requests for better performance and implements retries when the JWT has expired. You can run it like the following:

```
python3 examples/interactive_query.py -u https://my_org-my_account.us-west-2.aws.snowflakecomputing.com -s DB.SCHEMA.SERVICE_NAME -c "description,text" -a my_account -k /path/to/my/rsa_key.p8 -n my_name
```

This will launch an interactive session, where you will be prompted repeatedly for search queries to your Cortex Search Service.

### 3. Command line usage (cURL)

First, generate a JWT. For instance, if you have a private RSA key at the relative path `rsa_key.p8`, you can run the following from a shell (passing your account and user):

`snowsql --private-key-path rsa_key.p8 --generate-jwt -a my_org-my_account -u my_name`

Then, export the following variables in your shell session:

```
export CORTEX_SEARCH_JWT=<JWT here>
export CORTEX_SEARCH_DATABASE=MY_DB
export CORTEX_SEARCH_SCHEMA=MY_SCHEMA
export CORTEX_SEARCH_SERVICE_NAME=MY_SERVICE_NAME
export CORTEX_SEARCH_BASE_URL='https://my_org-my_account.us-west-2.aws.snowflakecomputing.com'
```

Then, you can run the following cURL command (modifiying the `data` passed as needed):

```
curl --location "$CORTEX_SEARCH_BASE_URL/api/v2/databases/$CORTEX_SEARCH_DATABASE/schemas/$CORTEX_SEARCH_SCHEMA/cortex-search-services/$CORTEX_SEARCH_SERVICE_NAME:query" \
--header 'X-Snowflake-Authorization-Token-Type: KEYPAIR_JWT' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header "Authorization: Bearer $CORTEX_SEARCH_JWT" \
--data '{
   "query": the sky is blue",
   "columns": ["description", "text"],
   "limit": 10
}'
```

## License

Apache Version 2.0
