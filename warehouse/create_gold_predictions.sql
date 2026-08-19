-- Gold prediction tables for Snowflake. Run this ONCE in a worksheet before
-- warehouse/load_gold_predictions_to_snowflake.py (the loader uses
-- auto_create_table=False, so these must exist first).
--
-- Populated from the Delta tables written by serving/spark_predict_and_gold.py:
--   gold/model_predictions -> GOLD_MODEL_PREDICTIONS  (every scored flow)
--   gold/security_alerts   -> GOLD_SECURITY_ALERTS    (predicted_class != Benign)
--
-- Timestamps (matching SILVER_FLOWS' TIMESTAMP_NTZ convention):
--   flow_timestamp       = when the flow happened (the SOC-timeline time) -- use
--                          THIS for alerts-over-time visuals so they align with
--                          the traffic pages.
--   prediction_timestamp = when the model scored it (~load time).

USE WAREHOUSE NIDD_WH;
USE DATABASE NIDD_DB;
USE SCHEMA SILVER;

CREATE TABLE IF NOT EXISTS GOLD_MODEL_PREDICTIONS (
    event_id             VARCHAR(64),
    predicted_class      VARCHAR(32),
    probability          FLOAT,
    model_version        VARCHAR(32),
    prediction_timestamp TIMESTAMP_NTZ,
    flow_timestamp       TIMESTAMP_NTZ,
    loaded_at            TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS GOLD_SECURITY_ALERTS (
    event_id             VARCHAR(64),
    predicted_class      VARCHAR(32),
    probability          FLOAT,
    model_version        VARCHAR(32),
    severity             VARCHAR(16),
    prediction_timestamp TIMESTAMP_NTZ,
    flow_timestamp       TIMESTAMP_NTZ,
    loaded_at            TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
