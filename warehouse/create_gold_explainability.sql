-- Gold explainability + drift tables for Snowflake / Power BI (Page 3).
-- Populated by modeling/src/export_shap_global.py and export_drift.py.

USE WAREHOUSE NIDD_WH;
USE DATABASE NIDD_DB;
USE SCHEMA SILVER;

-- Global SHAP feature importance: mean |SHAP| per feature, per class, plus an
-- 'Overall' class row set (mean across classes). Powers the "SHAP global" and
-- "top features per attack" visuals. (RANK is a Snowflake reserved word, hence
-- FEATURE_RANK.)
CREATE TABLE IF NOT EXISTS GOLD_SHAP_GLOBAL (
    model_version VARCHAR(32),
    class         VARCHAR(32),
    feature       VARCHAR(128),
    mean_abs_shap FLOAT,
    feature_rank  NUMBER(38,0)
);

-- Data-drift per feature: current (live/val) distribution vs the training
-- reference. PSI (population stability index) + KS test, with a denormalized
-- overall verdict on every row.
CREATE TABLE IF NOT EXISTS GOLD_DRIFT (
    feature                 VARCHAR(128),
    psi                     FLOAT,
    ks_statistic            FLOAT,
    ks_p_value              FLOAT,
    drifted                 NUMBER(1,0),
    share_drifted           FLOAT,
    dataset_drift_detected  NUMBER(1,0),
    reference_size          NUMBER(38,0),
    current_size            NUMBER(38,0),
    computed_at             TIMESTAMP_NTZ
);
