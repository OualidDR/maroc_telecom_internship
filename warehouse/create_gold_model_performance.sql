-- Gold model-performance table for Snowflake / Power BI (Page 4).
-- One row per class, with the model's offline precision/recall/F1 on the
-- held-out TEST set, plus denormalized overall metrics (accuracy, macro-F1,
-- weighted-F1) repeated on every row so Power BI cards can read them directly.
-- Populated by modeling/src/export_model_performance.py.
--
-- Note: PRECISION is an ANSI reserved word in Snowflake, so the columns are
-- named PRECISION_SCORE / RECALL_SCORE to stay safe.

USE WAREHOUSE NIDD_WH;
USE DATABASE NIDD_DB;
USE SCHEMA SILVER;

CREATE TABLE IF NOT EXISTS GOLD_MODEL_PERFORMANCE (
    model_version   VARCHAR(32),
    class           VARCHAR(32),
    precision_score FLOAT,
    recall_score    FLOAT,
    f1_score        FLOAT,
    support         NUMBER(38,0),
    accuracy        FLOAT,
    macro_f1        FLOAT,
    weighted_f1     FLOAT,
    evaluated_at    TIMESTAMP_NTZ
);
