# Reranker Finetuning 

This tutorial notebook guides you through finetuning a reranker model on your collection of documents, creating a service and use the service in your search workflow.

## When do you need to finetune a reranker?

Generally speaking, you can benefit from a finetuned reranker when you frequently observe irrelevant documents ranked in top positions by Cortex Search. This might happen when:
- Your search task is quite different from the standard "short query, long document" search format.
- Your search task requires an understanding of technical, proprietary terms, concepts, and jargon that are rarely found on the open web.
- Your search task involves languages that Cortex Search is not optimized for.

## Prerequisites

- Snowflake account with SPCS enabled
- Appropriate Snowflake privileges
  - `CREATE DATABASE`, `CREATE SCHEMA`
  - `CREATE COMPUTE POOL`
  - `CREATE IMAGE REPOSITORY`, `CREATE SERVICE`

**This tutorial also relies on a preview feature.** Before you can proceed with this tutorial, reach out to your account team to ask to enable this feature for your account:
  - [Batch Cortex Search](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/batch-cortex-search)


## Usage
- Create a compute pool with GPU resources following the [instructions here](https://docs.snowflake.com/en/sql-reference/sql/create-compute-pool)
- Upload the [attached notebook](../10_reranker_finetuning/reranker_finetuning.ipynb) to Snowflake using the [instructions here](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-create#create-a-new-notebook). Make sure to use `Snowflake ML Runtime GPU 1.0`, select the compute pool created, and turn on [all external access options](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-external-access) for downloading Python packages and base models.
- Follow the instructions in the notebook for data processing, prompt generation, synthetic data generation, training and deployment.