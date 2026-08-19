"""
serving/spark_predict_and_gold.py
====================================
Reads Silver as a Delta STREAM (not a fresh Kafka subscription -- Delta
tables support streaming reads, so this reuses what streaming/spark_bronze_silver.py
already wrote instead of duplicating Kafka consumption). Each micro-batch is
sent to the DS colleague's /predict/raw (or mocked, see predict_client.py),
joined back onto the flow, and split into two Gold outputs:

  - gold/model_predictions : every scored flow (event_id, predicted_class,
    probability, model_version, prediction_timestamp)
  - gold/security_alerts   : the subset where predicted_class != Benign,
    with a simple severity heuristic (placeholder, refine with DS input)

This is a SEPARATE streaming job from spark_bronze_silver.py on purpose --
it depends on an external API that doesn't exist yet at the time of writing
(mock mode), so keeping it isolated means it can't break the already-working
Bronze/Silver pipeline if it misbehaves.

Usage:
    python3 serving/spark_predict_and_gold.py
"""

import os
import sys
from datetime import datetime, timezone

# On Windows, bare "python"/"python3" hits the Microsoft Store app-execution
# alias stub instead of the venv interpreter, so Spark's Python workers fail to
# launch ("Python was not found" / "Python worker failed to connect back").
# Pin PySpark to THIS interpreter (the venv one running the driver). Harmless
# on Linux/macOS. Must be set before the SparkSession is built.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from dotenv import load_dotenv

# Load .env BEFORE importing predict_client so PREDICT_API_MOCK / URL are visible.
# (predict_client now also re-reads them per call, so this is defense in depth.)
load_dotenv()

sys.path.append(".")
from serving.predict_client import call_predict_api  # noqa: E402

SILVER_PATH = "s3a://silver/flows"
GOLD_PREDICTIONS_PATH = "s3a://gold/model_predictions"
GOLD_ALERTS_PATH = "s3a://gold/security_alerts"
GOLD_PREDICTIONS_CHECKPOINT = "s3a://gold/_checkpoints/model_predictions"

# Explicit schemas -- deliberately NOT relying on Spark's type inference on
# a list-of-dicts, which proved fragile (produced an invalid empty-struct
# Parquet schema for prediction_timestamp on this environment). Every value
# is cast to a plain str/float in Python before hitting Spark, and
# prediction_timestamp is parsed to a real timestamp via to_timestamp()
# right after creation.
# flow_timestamp = the flow's own ingestion_timestamp (the synthetic SOC-timeline
# time set by the simulator), carried through so Gold alerts/predictions align
# with the traffic timeline on the dashboards. prediction_timestamp = when the
# model actually scored it (~now). Keeping both separates "when it happened" from
# "when we detected it".
PREDICTIONS_SCHEMA = StructType([
    StructField("event_id", StringType(), True),
    StructField("predicted_class", StringType(), True),
    StructField("probability", FloatType(), True),
    StructField("model_version", StringType(), True),
    StructField("prediction_timestamp", StringType(), True),
    StructField("flow_timestamp", StringType(), True),
])

ALERTS_SCHEMA = StructType([
    StructField("event_id", StringType(), True),
    StructField("predicted_class", StringType(), True),
    StructField("probability", FloatType(), True),
    StructField("model_version", StringType(), True),
    StructField("severity", StringType(), True),
    StructField("prediction_timestamp", StringType(), True),
    StructField("flow_timestamp", StringType(), True),
])

# Placeholder severity heuristic -- ask the DS colleague whether this
# threshold logic matches what he'd expect from a security standpoint,
# this was picked arbitrarily to have SOMETHING working end-to-end.
def _severity(probability: float) -> str:
    if probability >= 0.90:
        return "High"
    if probability >= 0.70:
        return "Medium"
    return "Low"


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("5g-nidd-predict-and-gold")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        # 2g (up from 1g): the first micro-batch of a fresh run collect()s the
        # whole Silver backlog (150k rows) to the driver and builds a few pandas
        # copies before scoring. 1g was tight; 2g gives headroom. Runs on the
        # host (16GB), not the Docker VM, so this doesn't compete with the containers.
        .config("spark.driver.memory", "2g")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.2.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def _foreach_batch(batch_df, batch_id: int):
    # DataFrame.isEmpty() is JVM-native (Catalyst) -- unlike batch_df.rdd.isEmpty(),
    # which spins up a Python RDD worker just for the emptiness check and was
    # crashing under memory pressure on Windows.
    if batch_df.isEmpty():
        return

    # Manual pandas conversion -- see gx_spark_validator.py for why
    # .toPandas() is avoided (PySpark 3.5.1 / Python 3.12 incompatibility).
    rows = batch_df.collect()
    batch_pdf = pd.DataFrame([r.asDict() for r in rows])

    predictions_pdf = call_predict_api(batch_pdf)
    if predictions_pdf.empty:
        print(f"[predict_and_gold] batch={batch_id}: no predictions returned, skipping write.")
        return

    now_iso = datetime.now(timezone.utc).isoformat()

    # Map event_id -> the flow's own ingestion_timestamp (the synthetic SOC-
    # timeline time), so we can stamp each prediction with WHEN THE FLOW
    # HAPPENED, not just when we scored it. Falls back to now if a flow somehow
    # lacks a timestamp.
    ts_by_event = {}
    if "ingestion_timestamp" in batch_pdf.columns:
        for eid, ts in zip(batch_pdf["event_id"].astype(str), batch_pdf["ingestion_timestamp"]):
            ts_by_event[eid] = ts

    def _flow_ts(event_id: str) -> str:
        ts = ts_by_event.get(event_id)
        try:
            if ts is None or pd.isna(ts):
                return now_iso
        except (TypeError, ValueError):
            pass
        return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    # Force plain Python str/float on every field, one row at a time --
    # no reliance on pandas dtypes or Spark's inference to get this right.
    predictions_records = [
        {
            "event_id": str(r["event_id"]),
            "predicted_class": str(r["predicted_class"]),
            "probability": float(r["probability"]),
            "model_version": str(r["model_version"]),
            "prediction_timestamp": now_iso,
            "flow_timestamp": _flow_ts(str(r["event_id"])),
        }
        for r in predictions_pdf.to_dict(orient="records")
    ]

    spark = batch_df.sparkSession

    # NOTE: same distutils/Python 3.12 issue as gx_spark_validator.py, but
    # in the OTHER direction here -- spark.createDataFrame(pandas_df) hits
    # the same buggy require_minimum_pandas_version() check that toPandas()
    # does. Passing a list of dicts + an explicit schema avoids that
    # pandas-specific code path entirely AND avoids relying on inference.
    predictions_sdf = (
        spark.createDataFrame(predictions_records, schema=PREDICTIONS_SCHEMA)
        .withColumn("prediction_timestamp", F.to_timestamp("prediction_timestamp"))
        .withColumn("flow_timestamp", F.to_timestamp("flow_timestamp"))
    )

    # gold_model_predictions: every scored row
    predictions_sdf.write.format("delta").mode("append").save(GOLD_PREDICTIONS_PATH)

    # gold_security_alerts: malicious predictions only, with severity
    alert_records = [
        {**r, "severity": _severity(r["probability"])}
        for r in predictions_records
        if r["predicted_class"] != "Benign"
    ]
    if alert_records:
        alerts_sdf = (
            spark.createDataFrame(alert_records, schema=ALERTS_SCHEMA)
            .withColumn("prediction_timestamp", F.to_timestamp("prediction_timestamp"))
            .withColumn("flow_timestamp", F.to_timestamp("flow_timestamp"))
        )
        alerts_sdf.write.format("delta").mode("append").save(GOLD_ALERTS_PATH)

    n_alerts = len(alert_records)
    print(f"[predict_and_gold] batch={batch_id} scored={len(predictions_records)} alerts={n_alerts}")


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # maxFilesPerTrigger bounds each micro-batch to a handful of Silver files
    # (~a few thousand rows) instead of reading the ENTIRE backlog (150k) in one
    # batch. A giant first batch made collect() + the Python worker run out of
    # memory (they compete with the API + MLflow containers), crashing the
    # worker. Small batches keep memory flat and give incremental progress /
    # checkpointing -- you'll see several `[predict_and_gold] batch=N ...` lines.
    silver_stream = (
        spark.readStream.format("delta")
        .option("maxFilesPerTrigger", 20)
        .load(SILVER_PATH)
    )

    # trigger(availableNow=True): process all currently-available Silver in
    # back-to-back micro-batches (bounded by maxFilesPerTrigger), then STOP.
    # This is a one-shot backfill of a fixed 150k backlog, not a live stream --
    # the old processingTime="15 seconds" idled 15s between every micro-batch,
    # which with hundreds of small files meant hours of pure waiting. availableNow
    # runs them with no idle gap and terminates on its own (no Ctrl+C needed).
    query = (
        silver_stream.writeStream
        .outputMode("append")
        .foreachBatch(_foreach_batch)
        .option("checkpointLocation", GOLD_PREDICTIONS_CHECKPOINT)
        .trigger(availableNow=True)
        .start()
    )

    print("Predict-and-Gold streaming started. Reading Silver ->", SILVER_PATH)
    print("Writing predictions ->", GOLD_PREDICTIONS_PATH)
    print("Writing alerts ->", GOLD_ALERTS_PATH)
    print("Press Ctrl+C to stop.")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()