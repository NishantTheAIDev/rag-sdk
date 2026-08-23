"""Serializing experiment results to CSV, JSON, HTML, and a leaderboard."""

# The embedded HTML template intentionally contains long lines.
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rag_sdk.config import RagConfig
from rag_sdk.config.loader import dump_config
from rag_sdk.experiments.records import ExperimentRecord, ExperimentResult

_METRICS = ["hit_at_k", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map"]

_CSV_COLUMNS = [
    "run_id",
    "chunking",
    "chunk_size",
    "retrieval",
    "embedding",
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "map",
    "latency_mean_ms",
    "latency_median_ms",
    "total_chunks",
    "skipped_parameters",
    "timestamp",
]


def _row(record: ExperimentRecord) -> dict[str, object]:
    config = record.config
    chunking = config["chunking"]
    return {
        "run_id": record.run_id,
        "chunking": chunking["strategy"],
        "chunk_size": chunking.get("chunk_size", ""),
        "retrieval": config["retrieval"]["strategy"],
        "embedding": config["embedding"]["provider"],
        "hit_at_k": record.metrics["hit_at_k"],
        "precision_at_k": record.metrics["precision_at_k"],
        "recall_at_k": record.metrics["recall_at_k"],
        "mrr": record.metrics["mrr"],
        "ndcg_at_k": record.metrics["ndcg_at_k"],
        "map": record.metrics["map"],
        "latency_mean_ms": record.latency_ms.mean_ms,
        "latency_median_ms": record.latency_ms.median_ms,
        "total_chunks": record.total_chunks,
        "skipped_parameters": "; ".join(record.skipped_parameters),
        "timestamp": record.timestamp.isoformat(),
    }


def write_csv(result: ExperimentResult, path: str | Path) -> None:
    """Write one row per run to a CSV file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for record in result.records:
            writer.writerow(_row(record))


def write_json(result: ExperimentResult, path: str | Path) -> None:
    """Write full records plus a leaderboard to a JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": result.dataset_path,
        "dataset_hash": result.dataset_hash,
        "primary_metric": result.primary_metric,
        "k": result.k,
        "recommended": _recommendation(result),
        "leaderboard": [
            _record_payload(record) for record in result.leaderboard()
        ],
        "records": [_record_payload(record) for record in result.records],
    }
    output.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def write_leaderboard(result: ExperimentResult, path: str | Path) -> None:
    """Write the leaderboard sorted by the primary metric."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["rank", "run_id", result.primary_metric, *_METRICS, "latency_mean_ms"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rank, record in enumerate(result.leaderboard(), start=1):
            row = {
                "rank": rank,
                "run_id": record.run_id,
                result.primary_metric: record.metrics[result.primary_metric],
            }
            for metric in _METRICS:
                row[metric] = record.metrics[metric]
            row["latency_mean_ms"] = record.latency_ms.mean_ms
            writer.writerow(row)


def write_html(result: ExperimentResult, path: str | Path) -> None:
    """Write a self-contained interactive HTML report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": result.dataset_path,
        "dataset_hash": result.dataset_hash,
        "primary_metric": result.primary_metric,
        "k": result.k,
        "metrics": _METRICS,
        "records": [_row(record) for record in result.records],
        "recommended": _recommendation(result),
    }
    data = json.dumps(payload)
    output.write_text(_HTML_TEMPLATE.replace("__DATA__", data), encoding="utf-8")


def write_reports(result: ExperimentResult, output_dir: str | Path) -> dict[str, Path]:
    """Write CSV, JSON, leaderboard, and HTML reports into ``output_dir``."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": directory / "results.csv",
        "json": directory / "results.json",
        "leaderboard": directory / "leaderboard.csv",
        "html": directory / "report.html",
    }
    write_csv(result, paths["csv"])
    write_json(result, paths["json"])
    write_leaderboard(result, paths["leaderboard"])
    write_html(result, paths["html"])
    return paths


def _recommendation(result: ExperimentResult) -> dict[str, Any]:
    best = result.leaderboard()[0]
    config = RagConfig.model_validate(best.config)
    return {
        "run_id": best.run_id,
        "primary_metric": result.primary_metric,
        "value": best.metrics[result.primary_metric],
        "config_yaml": dump_config(config).strip(),
    }


def _record_payload(record: ExperimentRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG SDK Experiment Report</title>
<style>
  :root { --accent: #4f46e5; --border: #e5e7eb; --bg: #f9fafb; }
  * { box-sizing: border-box; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: var(--bg); color: #111827; }
  main { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .sub { color: #6b7280; font-size: 0.875rem; margin-bottom: 1.5rem; }
  .banner { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.5rem; }
  .banner h2 { margin: 0 0 0.5rem; font-size: 1rem; color: var(--accent); }
  .banner pre { margin: 0; white-space: pre-wrap; font-size: 0.8rem; background: #fff; border: 1px solid var(--border); border-radius: 0.375rem; padding: 0.75rem; }
  table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid var(--border); border-radius: 0.5rem; overflow: hidden; font-size: 0.875rem; }
  th, td { padding: 0.5rem 0.625rem; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { background: #f3f4f6; cursor: pointer; user-select: none; position: sticky; top: 0; }
  th:hover { background: #e5e7eb; }
  th::after { content: " \\2195"; color: #9ca3af; font-size: 0.7rem; }
  tr.best { background: #ecfdf5; }
  .muted { color: #9ca3af; }
  .wrap { max-width: 160px; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>
<main>
<h1>RAG SDK Experiment Report</h1>
<div class="sub" id="meta"></div>
<div class="banner" id="banner"></div>
<table id="results">
  <thead><tr id="header"></tr></thead>
  <tbody id="body"></tbody>
</table>
</main>
<script>
const DATA = __DATA__;
const primary = DATA.primary_metric;
const metrics = DATA.metrics;
const rec = DATA.recommended;

document.getElementById("meta").textContent =
  "Dataset: " + DATA.dataset + "  (hash " + DATA.dataset_hash.slice(0, 12) + ")  |  k = " + DATA.k +
  "  |  primary metric: " + primary;

const banner = document.getElementById("banner");
banner.innerHTML =
  "<h2>Recommended configuration (" + primary + " = " + rec.value.toFixed(4) + ", " + rec.run_id + ")</h2>" +
  "<pre>" + rec.config_yaml + "</pre>";

const cols = ["run_id", "chunking", "chunk_size", "retrieval", "embedding"].concat(metrics, ["latency_mean_ms", "latency_median_ms", "skipped_parameters"]);
const header = document.getElementById("header");
cols.forEach(function (col) {
  const th = document.createElement("th");
  th.textContent = col;
  th.addEventListener("click", function () { sortBy(col); });
  header.appendChild(th);
});

const rows = DATA.records.map(function (r) {
  const flat = {};
  cols.forEach(function (col) { flat[col] = r[col]; });
  return flat;
});

function num(v) { return typeof v === "number"; }

function render() {
  const body = document.getElementById("body");
  body.innerHTML = "";
  rows.forEach(function (row) {
    const tr = document.createElement("tr");
    if (row.run_id === rec.run_id) tr.className = "best";
    cols.forEach(function (col) {
      const td = document.createElement("td");
      const v = row[col];
      td.textContent = num(v) ? v.toFixed(4) : v;
      if (col === "run_id") td.classList.add("wrap");
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

let sortCol = null;
let sortAsc = true;
function sortBy(col) {
  if (sortCol === col) { sortAsc = !sortAsc; } else { sortCol = col; sortAsc = true; }
  rows.sort(function (a, b) {
    const va = a[col], vb = b[col];
    if (num(va) && num(vb)) return sortAsc ? va - vb : vb - va;
    return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  });
  render();
}

render();
</script>
</body>
</html>
"""