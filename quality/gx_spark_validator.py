"""
quality/gx_spark_validator.py
===============================
Wires Great Expectations into the live Spark streaming job. Each Silver
micro-batch is validated against the SAME suite used by
quality/expectations_silver.py (not a duplicated set of rules).

Design choice: validation does NOT block the write. A failing batch is
still written to Silver (it's already been checked for the hard structural
stuff -- schema, types -- by Spark itself before it gets here), but the
validation OUTCOME is logged to a Delta table
(s3a://silver/_quality_log/gx_results) with one row per batch:
event_timestamp, batch_id, row_count, success, n_failed_expectations,
failed_expectation_summary.

Why log-not-block: this is a streaming pipeline, not a batch ETL job --
halting the whole stream because one micro-batch had a few out-of-range
values would be worse than the bad data itself (the DS side already treats
nulls/sentinels defensively). The log table becomes the first real
"quality over time" metric -- exactly the kind of series Grafana will
plot once Prometheus/Grafana are wired in (S4).

Stretch goal (not implemented): row-level quarantine, i.e. splitting each
batch into valid/invalid rows and writing invalid ones to a separate
s3a://silver/_quarantine/ table instead of just logging aggregate pass/fail.
"""

import sys
from datetime import datetime, timezone

import great_expectations as gx
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

sys.path.append(".")
from quality.expectations_silver import build_suite  # noqa: E402
from quality.gx_context import get_context  # noqa: E402

QUALITY_LOG_PATH = "s3a://silver/_quality_log/gx_results"


class SparkBatchValidator:
    """Built once when the stream starts, then called once per micro-batch.

    Building the GX context + suite + validation_definition is relatively
    expensive (disk IO, suite construction) -- doing it once per stream
    instead of once per batch keeps the 10s trigger interval realistic on
    a 2-core machine.
    """

    def __init__(self):
        # Ephemeral (in-memory) context, NOT the shared file-backed one from
        # quality/gx_context.py. This validator is rebuilt fresh every time
        # the Spark job starts, so there's no need (and no benefit) to
        # persist its datasource/asset config to disk -- doing so caused
        # "datasource already exists" errors on every restart, since the
        # file-backed context remembered objects from the previous run.
        self.context = gx.get_context(mode="ephemeral")
        self.suite = build_suite()
        self.context.suites.add_or_update(self.suite)

        data_source = self.context.data_sources.add_pandas(name="pandas_spark_batches")
        asset = data_source.add_dataframe_asset(name="silver_batch_df")
        self.batch_definition = asset.add_batch_definition_whole_dataframe(name="silver_batch")

        self.validation_definition = gx.ValidationDefinition(
            name="silver_batch_validation",
            data=self.batch_definition,
            suite=self.suite,
        )
        self.context.validation_definitions.add_or_update(self.validation_definition)

        print("SparkBatchValidator ready -- GX suite built once, reused per micro-batch.")

    def validate(self, batch_pdf: pd.DataFrame, batch_id: int) -> dict:
        """Run the suite against one micro-batch, return a summary dict."""
        result = self.validation_definition.run(batch_parameters={"dataframe": batch_pdf})

        failed = [r for r in result.results if not r.success]
        failed_summary = "; ".join(
            f"{r.expectation_config.type}({r.expectation_config.kwargs.get('column', '?')})"
            for r in failed[:10]  # cap the string, don't blow up the log row
        )

        return {
            "event_timestamp": datetime.now(timezone.utc),
            "batch_id": int(batch_id),
            "row_count": int(len(batch_pdf)),
            "success": bool(result.success),
            "n_failed_expectations": len(failed),
            "failed_expectation_summary": failed_summary,
        }


def make_foreach_batch_fn(validator: SparkBatchValidator, spark: SparkSession, silver_path: str):
    """Returns the function passed to .foreachBatch() on the Silver writeStream."""

    def _foreach_batch(batch_df: SparkDataFrame, batch_id: int):
        if batch_df.rdd.isEmpty():
            return

        # Always write the batch to Silver -- validation informs, doesn't gate.
        batch_df.write.format("delta").mode("append").partitionBy("attack_type").save(silver_path)

        # NOTE: intentionally NOT using batch_df.toPandas() -- PySpark 3.5.1's
        # toPandas() internally does `from distutils.version import
        # LooseVersion`, and distutils was removed from the standard library
        # in Python 3.12, so toPandas() raises ModuleNotFoundError there.
        # Manual conversion sidesteps PySpark's version-check code path
        # entirely. Batches are small (maxOffsetsPerTrigger=500), so the
        # lack of Arrow-accelerated conversion doesn't matter here.
        rows = batch_df.collect()
        batch_pdf = pd.DataFrame([r.asDict() for r in rows])
        summary = validator.validate(batch_pdf, batch_id)

        status = "OK" if summary["success"] else f"FAILED ({summary['n_failed_expectations']} expectations)"
        print(f"[GX] batch={batch_id} rows={summary['row_count']} -> {status}")
        if not summary["success"]:
            print(f"     {summary['failed_expectation_summary']}")

        log_row = spark.createDataFrame([summary])
        log_row.write.format("delta").mode("append").save(QUALITY_LOG_PATH)

    return _foreach_batch