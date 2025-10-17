# Reranker Finetuning 

This tutorial notebook guides you through finetuning a reranker model on your collection of documents, creating a service and use the service in your search workflow.

## Prerequisites
**This tutorial relies on a preview feature.** Before you can proceed with this tutorial, reach out to your account team to ask to enable this feature for your account:
  - [Batch Cortex Search](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/batch-cortex-search)

**Python Package Requirements:**
  - Snowflake Python API version 1.6.0 or later (required for multi-index query syntax)

## Usage
- Create a compute pool with GPU resources following the [instructions here](https://docs.snowflake.com/en/sql-reference/sql/create-compute-pool)
- Upload the [attached notebook](../10_reranker_finetuning/reranker_finetuning.ipynb) to Snowflake using the [instructions here](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-create#create-a-new-notebook). Make sure to use `Snowflake ML Runtime GPU 1.0`, select the compute pool created, and turn on external access (PYPI and HF).
- Follow the instructions in the notebook for data processing, prompt generation, synthetic data generation, training and deployment.