#!/bin/bash
# reset_pipeline.sh
# ===================
# Wipes Kafka topic + MinIO (Bronze, Silver, checkpoints, quality log) so
# you can re-run the whole pipeline from a clean slate. Run this BEFORE
# starting spark_bronze_silver.py + replay_simulator.py for a fresh test.
#
# Usage:
#   chmod +x reset_pipeline.sh
#   ./reset_pipeline.sh

set -e

echo "== 1/2 Resetting Kafka topic 'flows-raw' =="
docker exec -it redpanda rpk topic delete flows-raw || echo "  (topic didn't exist yet, that's fine)"
docker exec -it redpanda rpk topic create flows-raw --partitions 3

echo ""
echo "== 2/2 Wiping MinIO: Bronze, Silver, Gold, checkpoints, quality log =="
# NB: Gold (predictions/alerts + its checkpoint) MUST be wiped too. Otherwise
# old predictions (incl. earlier mock-mode runs) mix with the new ones, and the
# stale gold checkpoint still points at the now-deleted Silver table -- which
# makes serving/spark_predict_and_gold.py fail on restart with a Delta version
# mismatch. Wiping mlflow-artifacts is intentionally NOT done here: the
# registered model lives there and must survive a pipeline reset.
docker compose run --rm --entrypoint sh minio-init -c "
mc alias set local http://minio:9000 \${MINIO_ROOT_USER:-minioadmin} \${MINIO_ROOT_PASSWORD:-minioadmin};
mc rm --recursive --force local/bronze/flows 2>/dev/null || true;
mc rm --recursive --force local/bronze/_checkpoints 2>/dev/null || true;
mc rm --recursive --force local/silver/flows 2>/dev/null || true;
mc rm --recursive --force local/silver/_checkpoints 2>/dev/null || true;
mc rm --recursive --force local/silver/_quality_log 2>/dev/null || true;
mc rm --recursive --force local/gold/model_predictions 2>/dev/null || true;
mc rm --recursive --force local/gold/security_alerts 2>/dev/null || true;
mc rm --recursive --force local/gold/_checkpoints 2>/dev/null || true;
mc mb --ignore-existing local/bronze;
mc mb --ignore-existing local/silver;
mc mb --ignore-existing local/gold;
echo 'MinIO wiped and buckets ready.';
"

echo ""
echo "== Reset complete. Pipeline is clean. =="
echo "Now run, in order:"
echo "  1) python3 streaming/spark_bronze_silver.py"
echo "  2) (wait for 'Streaming started...')"
echo "  3) python3 ingestion/replay_simulator.py --rate 200 --limit 2000"
