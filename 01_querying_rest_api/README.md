# Querying the Cortex Search REST API

## Prerequisites

Before you can run the script, ensure you have the following prerequisites installed:

- Python 3.x
- pip (Python package installer)
  Additionally, you must have access to a Snowflake account and the required permissions to query the Cortex Search Service at the specified database and schema.

Installation
First, clone this repository to your local machine using git:

```
git clone https://github.com/snowflake-labs/cortex-search.git
cd cortex-search/01_querying_rest_api
```

Install the necessary Python dependencies by running:

```
pip install -r requirements.txt
```

## Key-pair auth configuration

Additionally, you must generate a private key for JWT auth with Snowflake as described in [this document](https://docs.snowflake.com/user-guide/key-pair-auth#configuring-key-pair-authentication).

**Note**: take note of the path to your generated RSA private key, e.g., `/path/to/my/rsa_key.p8` -- you will need to supply this as the `--private-key-path` parameter to query the service later from the command line, or list the path to the file from within a notebook.

## Command line usage

The `simple_query.py` example script can be executed from the command line. For instance:

```
python3 examples/simple_query.py -u https://my_org-my_account.us-west-2.aws.snowflakecomputing.com -s DB.SCHEMA.SERVICE_NAME -q "the sky is blue" -c "description,text" -l 10 -a my_account -k /path/to/my/rsa_key.p8 -n my_name
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

## License

Apache Version 2.0
