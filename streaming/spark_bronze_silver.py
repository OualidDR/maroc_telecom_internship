"""
streaming/spark_bronze_silver.py
==================================
Spark Structured Streaming job:
  Kafka/Redpanda topic "flows-raw"
    -> BRONZE  (s3a://bronze/flows)  raw JSON payload as received, untouched
    -> SILVER  (s3a://silver/flows)  typed/cleaned according to contracts/schemas.py

Sized for a low-resource laptop: local[2], small shuffle partitions, small
Kafka maxOffsetsPerTrigger. Don't run this alongside a heavy Kafka replay
rate on the same machine -- start with a modest --rate on the simulator.

Usage:
    python3 streaming/spark_bronze_silver.py

Prereqs (see README section this script was generated with):
    pip install pyspark==3.5.1 delta-spark==3.2.0
    A JDK (11 or 17) must be installed and JAVA_HOME set.
    docker compose up -d   (redpanda + minio must be running)
"""

import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, IntegerType
)

sys.path.append(".")
from contracts.schemas import FEATURE_SCHEMA, LABEL_COLUMNS  # noqa: E402

# -----------------------------------------------------------------------------
# Config -- adjust here if you move to a bigger machine later
# -----------------------------------------------------------------------------
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "flows-raw"

MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

BRONZE_PATH = "s3a://bronze/flows"
SILVER_PATH = "s3a://silver/flows"
BRONZE_CHECKPOINT = "s3a://bronze/_checkpoints/flows"
SILVER_CHECKPOINT = "s3a://silver/_checkpoints/flows"

MAX_OFFSETS_PER_TRIGGER = 700  # small batches, gentle on 2 cores / limited RAM


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("5g-nidd-bronze-silver")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
            "io.delta:delta-spark_2.12:3.2.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # MinIO / S3A wiring
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


# -----------------------------------------------------------------------------
# Silver-layer schema, generated FROM the shared contract -- not hand-duplicated
# -----------------------------------------------------------------------------
_DTYPE_MAP = {
    "float": DoubleType(),
    "int": LongType(),
    "category": StringType(),
}


def build_flow_schema() -> StructType:
    fields = [StructField(f.name, _DTYPE_MAP[f.dtype], nullable=f.nullable) for f in FEATURE_SCHEMA]
    fields += [StructField(name, StringType(), nullable=False) for name in LABEL_COLUMNS]
    fields += [
        StructField("is_tcp", IntegerType(), nullable=False),
        StructField("has_dst_reply", IntegerType(), nullable=False),
    ]
    return StructType(fields)


EVENT_SCHEMA = StructType([
    StructField("event_id", StringType(), nullable=False),
    StructField("ingestion_timestamp", StringType(), nullable=False),
    StructField("schema_version", StringType(), nullable=False),
    StructField("flow", build_flow_schema(), nullable=False),
])


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
        .load()
    )

    # ---------------------------------------------------------------------
    # BRONZE: raw payload as received, only Kafka metadata added. No parsing,
    # no dropping, no typing -- this is the immutable "what we actually
    # received" layer, useful for replay/debugging if Silver logic changes.
    # ---------------------------------------------------------------------
    bronze_df = raw_stream.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.col("value").cast("string").alias("raw_json"),
        F.col("topic"),
        F.col("partition"),
        F.col("offset"),
        F.col("timestamp").alias("kafka_timestamp"),
    )

    bronze_query = (
        bronze_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", BRONZE_CHECKPOINT)
        .trigger(processingTime="10 seconds")
        .start(BRONZE_PATH)
    )

    # ---------------------------------------------------------------------
    # SILVER: parsed, typed, and flattened according to contracts/schemas.py.
    # Nulls are left as real nulls (structural, per the contract) -- no
    # imputation happens here, that's a modeling-time decision for the DS side.
    # ---------------------------------------------------------------------
    parsed_df = raw_stream.select(
        F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("event")
    )

    silver_df = parsed_df.select(
        F.col("event.event_id").alias("event_id"),
        F.to_timestamp(F.col("event.ingestion_timestamp")).alias("ingestion_timestamp"),
        F.col("event.schema_version").alias("schema_version"),
        "event.flow.*",
    )

    silver_query = (
        silver_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", SILVER_CHECKPOINT)
        .trigger(processingTime="10 seconds")
        .partitionBy("Attack Type")
        .start(SILVER_PATH)
    )

    print("Streaming started. Bronze ->", BRONZE_PATH, " Silver ->", SILVER_PATH)
    print("Press Ctrl+C to stop.")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
