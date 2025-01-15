# Cortex Search E-commerce Examples

This repository contains companion notebooks for the blog post "Improving E-commerce Search: Intelligent Ranking Made Simple with Snowflake Cortex Search" (link coming soon). These notebooks demonstrate how to set up, evaluate, and optimize Cortex Search for e-commerce use cases.

## Notebooks Overview

### 1. `wands_ingest.ipynb`
This notebook shows how to ingest and prepare the WANDS (Wayfair Product Search) dataset for use with Cortex Search:
- Downloads the WANDS dataset
- Processes product features and creates a unified text column for search
- Creates and configures a Cortex Search service with appropriate attributes

### 2. `soft_query_boost_example.ipynb`
Demonstrates the implementation of smart query boosting in Cortex Search with example feature extraction strategy:
- Shows how to extract main terms from search queries
- Implements soft boost logic for queries with supplemental signals
- Provides examples of query processing for products with multiple aspects

### 3. `llm_judge_product_search.ipynb`
Illustrates the query label extraction  process using LLM-based relevance judgments:
- Implements structured prompts for LLM-based relevance evaluation
- Shows how to process and evaluate search results
- Demonstrates integration with Snowflake's Cortex Search service

## Prerequisites

- Snowflake account with access to Cortex Search
- Python 3.11+
- Required Python packages:
  - snowflake-snowpark-python
  - pandas
  - numpy

## Setup Instructions

1. Clone this repository
2. Set up your Snowflake connection:
   ```python
   from snowflake.snowpark.context import get_active_session
   session = get_active_session()
   ```
3. Follow the notebooks in this recommended order:
   - Start with `wands_ingest.ipynb` to set up your data
   - Explore `soft_query_boost_example.ipynb` for query optimization
   - Use `llm_judge_product_search.ipynb` for automated relevance tuning

## Data Requirements

The notebooks work with several e-commerce datasets:
- [WANDS (Wayfair Product Search Dataset)](https://github.com/wayfair/WANDS)
- [TREC Product Search 2023](https://arxiv.org/abs/2311.07861)
- [TREC Product Search 2024](https://trec-product-search.github.io/index.html)
- [Amazon ESCI](https://github.com/amazon-research/esci-data)

Note: Some datasets may require separate download and setup. Please refer to the individual dataset documentation for access instructions.

## Additional Resources

- [Cortex Search Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/overview-tutorials)
- [Cortex Search Technical Blog](https://www.snowflake.com/engineering-blog/cortex-search-and-retrieval-enterprise-ai/)
- [Cortex Search Optimization App](../../examples/streamlit-evaluation/)

## License

Please refer to the license terms for individual datasets and Snowflake's terms of service for Cortex Search usage.
