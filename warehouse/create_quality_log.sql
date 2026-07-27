-- Run once in a Snowflake worksheet, in the SILVER schema (same as SILVER_FLOWS).

USE WAREHOUSE NIDD_WH;
USE DATABASE NIDD_DB;
USE SCHEMA SILVER;

CREATE TABLE IF NOT EXISTS QUALITY_LOG (
    event_timestamp             TIMESTAMP_NTZ NOT NULL,
    batch_id                    NUMBER(38,0) NOT NULL,
    row_count                   NUMBER(38,0) NOT NULL,
    success                     BOOLEAN NOT NULL,
    n_failed_expectations       NUMBER(38,0) NOT NULL,
    failed_expectation_summary  VARCHAR(2000),
    loaded_at                   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
