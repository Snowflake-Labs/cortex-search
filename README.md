# Cortex Search

This repository contains example usage, including authentication, for the Cortex Search REST API.

## Prerequisites

Before you can run the script, ensure you have the following prerequisites installed:

- Python 3.x
- pip (Python package installer)
  Additionally, you must have access to a Snowflake account and the required permissions to query the Cortex Search Service at the specified database and schema.

Installation
First, clone this repository to your local machine using git:

```
git clone https://github.com/snowflake-labs/cortex-search.git
cd cortex-search
```

Install the necessary Python dependencies by running:

```
pip install -r requirements.txt
```

## Usage

The `make_query.py` example script can be executed from the command line. For instance:

```
python3 examples/make_query.py -u https://my_org-my_account.us-west-2.aws.snowflakecomputing.com -s DB.SCHEMA.SERVICE_NAME -q "the sky is blue" -c "description,text" -l 10 -a my_account -k /path/to/my/rsa_key.p8 -n my_name
```

Arguments:

- -u, --url: URL of the Snowflake instance
- -s, --qualified-service-name: The service name in the format DATABASE.SCHEMA.SERVICE
- -q, --query: The search query term
- -c, --columns: Comma-separated list of columns to return in the results
- -l, --limit: The max number of results to return
- -f, --filter: (optional) Filter string
- -a, --account: Snowflake account name
- -k, --private-key-path: Path to the RSA private key file for authentication
- -n, --user-name: Username for the Snowflake account

## License

Apache Version 2.0
