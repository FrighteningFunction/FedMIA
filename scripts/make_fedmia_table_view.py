from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    REPO_ROOT / "aggregated_report" / "binn_fedmia_results.csv",
    REPO_ROOT / "aggregated_report" / "cifar100_alexnet_fedmia_results.csv",
]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "aggregated_report"

PAPER_COLUMNS = [
    "dataset_model",
    "config_label",
    "aggregation",
    "method",
    "member_count",
    "nonmember_count",
    "train_acc",
    "holdout_acc",
    "test_acc",
    "auc",
    "f1",
    "tpr",
    "tnr",
    "fpr",
    "threshold_accuracy",
    "tpr_at_fpr_0.001",
    "tpr_at_fpr_0.01",
    "tpr_at_fpr_0.1",
    "best_f1",
    "best_threshold",
    "source_file",
]

CONFIG_COLUMNS = [
    "dataset_model",
    "config_label",
    "aggregation",
    "method",
]

SPLIT_RESULT_COLUMNS = [
    [
        "member_count",
        "nonmember_count",
        "train_acc",
        "holdout_acc",
        "test_acc",
        "auc",
        "f1",
        "threshold_accuracy",
    ],
    [
        "tpr",
        "tnr",
        "fpr",
        "tpr_at_fpr_0.001",
        "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.1",
        "best_f1",
        "best_threshold",
    ],
]

SHORT_LABELS = {
    "dataset_model": "Dataset / Model",
    "config_label": "Configuration",
    "aggregation": "Aggregation",
    "method": "Method",
    "member_count": "Members",
    "nonmember_count": "Nonmembers",
    "train_acc": "Train Acc.",
    "holdout_acc": "Holdout Acc.",
    "test_acc": "Test Acc.",
    "auc": "AUC",
    "f1": "F1",
    "tpr": "TPR",
    "tnr": "TNR",
    "fpr": "FPR",
    "threshold_accuracy": "Threshold Acc.",
    "tpr_at_fpr_0.001": "TPR@FPR 0.001",
    "tpr_at_fpr_0.01": "TPR@FPR 0.01",
    "tpr_at_fpr_0.1": "TPR@FPR 0.1",
    "best_f1": "Best F1",
    "best_threshold": "Best Threshold",
    "source_file": "Source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Word-friendly interactive view over aggregated FedMIA results."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Aggregate CSV to include. Can be repeated. Defaults to BINN and CIFAR aggregate CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated HTML/CSV/TSV view files.",
    )
    parser.add_argument(
        "--columns",
        default=",".join(PAPER_COLUMNS),
        help="Comma-separated columns selected by default in the paper view.",
    )
    return parser.parse_args()


def read_rows(paths: Sequence[Path]) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    columns: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for column in reader.fieldnames or []:
                if column not in columns:
                    columns.append(column)
            for row in reader:
                row["_source_path"] = str(path)
                rows.append(row)
    return rows, columns


def write_delimited(path: Path, rows: Sequence[Dict[str, str]], columns: Sequence[str], delimiter: str):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def unique_values(rows: Iterable[Dict[str, str]], column: str) -> List[str]:
    return sorted({row.get(column, "") for row in rows if row.get(column, "")})


def render_options(values: Sequence[str]) -> str:
    options = ['<option value="">All</option>']
    for value in values:
        escaped = html.escape(value, quote=True)
        options.append(f'<option value="{escaped}">{escaped}</option>')
    return "\n".join(options)


def make_html(
    rows: Sequence[Dict[str, str]],
    columns: Sequence[str],
    default_columns: Sequence[str],
) -> str:
    labels = {column: SHORT_LABELS.get(column, column) for column in columns}
    data_json = json.dumps(rows, ensure_ascii=False)
    columns_json = json.dumps(list(columns))
    default_columns_json = json.dumps([column for column in default_columns if column in columns])
    config_columns_json = json.dumps([column for column in CONFIG_COLUMNS if column in columns])
    split_columns_json = json.dumps(
        [[column for column in group if column in columns] for group in SPLIT_RESULT_COLUMNS]
    )
    labels_json = json.dumps(labels)
    dataset_options = render_options(unique_values(rows, "dataset_model"))
    aggregation_options = render_options(unique_values(rows, "aggregation"))
    method_options = render_options(unique_values(rows, "method"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FedMIA Paper Table View</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1b1f23;
      --muted: #626a73;
      --line: #d7dce0;
      --accent: #245b9d;
      --accent-soft: #e7f0fb;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 18px;
      font-weight: 650;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(3, minmax(160px, 1fr)) minmax(260px, 2fr);
      gap: 10px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    select, input {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      align-items: center;
    }}
    details {{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfc;
    }}
    summary {{
      padding: 8px 10px;
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
    }}
    .column-panel {{
      padding: 0 10px 10px;
    }}
    .column-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .columns {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 5px 12px;
      max-height: 180px;
      overflow: auto;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }}
    .column-choice {{
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--ink);
      font-size: 12px;
    }}
    .column-choice input {{
      min-height: auto;
    }}
    button {{
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 7px 10px;
      font: inherit;
      cursor: pointer;
    }}
    button.secondary {{
      background: white;
      color: var(--accent);
    }}
    .status {{
      color: var(--muted);
      margin-left: 4px;
    }}
    main {{
      padding: 16px 24px 28px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      max-height: calc(100vh - 180px);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 1500px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: #eef1f3;
      position: sticky;
      top: 0;
      z-index: 2;
      text-align: left;
      font-weight: 650;
    }}
    tr.selected td {{
      background: var(--accent-soft);
    }}
    td.source {{
      max-width: 300px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .select-col {{
      width: 34px;
      text-align: center;
    }}
    @media (max-width: 900px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
      header {{
        position: static;
      }}
      .table-wrap {{
        max-height: none;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>FedMIA Paper Table View</h1>
    <div class="controls">
      <label>Dataset / model
        <select id="datasetFilter">{dataset_options}</select>
      </label>
      <label>Aggregation
        <select id="aggregationFilter">{aggregation_options}</select>
      </label>
      <label>Method
        <select id="methodFilter">{method_options}</select>
      </label>
      <label>Search
        <input id="searchFilter" type="search" placeholder="Filter by config, source, beta, metric value...">
      </label>
    </div>
    <div class="actions">
      <button id="selectVisible" class="secondary">Select Visible</button>
      <button id="clearSelection" class="secondary">Clear Selection</button>
      <button id="copySelected">Copy Selected TSV</button>
      <button id="copyVisible" class="secondary">Copy Visible TSV</button>
      <button id="downloadSelectedCsv" class="secondary">Download Selected CSV</button>
      <button id="copySplitA" class="secondary">Copy Split A TSV</button>
      <button id="copySplitB" class="secondary">Copy Split B TSV</button>
      <span id="status" class="status"></span>
    </div>
    <details>
      <summary>Choose columns</summary>
      <div class="column-panel">
        <div class="column-actions">
          <button id="usePaperColumns" class="secondary">Paper Default</button>
          <button id="useAllColumns" class="secondary">All Columns</button>
          <button id="useMainMetrics" class="secondary">Main Metrics</button>
          <button id="useLowFprMetrics" class="secondary">Low-FPR Metrics</button>
          <button id="clearColumns" class="secondary">Clear Columns</button>
        </div>
        <div id="columnChooser" class="columns"></div>
      </div>
    </details>
  </header>
  <main>
    <div class="table-wrap">
      <table id="resultsTable"></table>
    </div>
  </main>
  <script>
    const rows = {data_json};
    const columns = {columns_json};
    const defaultColumns = {default_columns_json};
    const configColumns = {config_columns_json};
    const splitColumns = {split_columns_json};
    const labels = {labels_json};
    const selected = new Set();
    const activeColumns = new Set(defaultColumns);

    const table = document.getElementById("resultsTable");
    const status = document.getElementById("status");
    const columnChooser = document.getElementById("columnChooser");
    const filters = {{
      dataset: document.getElementById("datasetFilter"),
      aggregation: document.getElementById("aggregationFilter"),
      method: document.getElementById("methodFilter"),
      search: document.getElementById("searchFilter"),
    }};

    function matches(row) {{
      if (filters.dataset.value && row.dataset_model !== filters.dataset.value) return false;
      if (filters.aggregation.value && row.aggregation !== filters.aggregation.value) return false;
      if (filters.method.value && row.method !== filters.method.value) return false;
      const query = filters.search.value.trim().toLowerCase();
      if (!query) return true;
      return columns.some(column => String(row[column] || "").toLowerCase().includes(query));
    }}

    function visibleIndexes() {{
      return rows.map((row, index) => [row, index]).filter(([row]) => matches(row)).map(([, index]) => index);
    }}

    function currentColumns() {{
      return columns.filter(column => activeColumns.has(column));
    }}

    function escapeText(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }}[char]));
    }}

    function render() {{
      const indexes = visibleIndexes();
      const shownColumns = currentColumns();
      const header = [
        '<thead><tr><th class="select-col">Use</th>',
        ...shownColumns.map(column => `<th>${{escapeText(labels[column] || column)}}</th>`),
        '</tr></thead>'
      ].join("");
      const body = indexes.map(index => {{
        const row = rows[index];
        const checked = selected.has(index) ? "checked" : "";
        const selectedClass = selected.has(index) ? " class=\\"selected\\"" : "";
        const cells = shownColumns.map(column => {{
          const cls = column === "source_file" ? ' class="source"' : "";
          return `<td${{cls}}>${{escapeText(row[column] || "")}}</td>`;
        }}).join("");
        return `<tr${{selectedClass}}><td class="select-col"><input type="checkbox" data-index="${{index}}" ${{checked}}></td>${{cells}}</tr>`;
      }}).join("");
      table.innerHTML = header + `<tbody>${{body}}</tbody>`;
      table.querySelectorAll('input[type="checkbox"][data-index]').forEach(box => {{
        box.addEventListener("change", event => {{
          const index = Number(event.target.dataset.index);
          if (event.target.checked) selected.add(index);
          else selected.delete(index);
          render();
        }});
      }});
      status.textContent = `${{indexes.length}} visible row(s), ${{selected.size}} selected, ${{shownColumns.length}} column(s)`;
      renderColumnChooser();
    }}

    function rowsToDelimited(indexes, selectedColumns, delimiter) {{
      const escapeTsv = value => String(value ?? "").replace(/\\t/g, " ").replace(/\\r?\\n/g, " ");
      const escapeCsv = value => {{
        const clean = String(value ?? "").replace(/\\r?\\n/g, " ");
        return /[",\\n]/.test(clean) ? '"' + clean.replace(/"/g, '""') + '"' : clean;
      }};
      const escapeValue = delimiter === "\\t" ? escapeTsv : escapeCsv;
      const header = selectedColumns.map(column => labels[column] || column).map(escapeValue).join(delimiter);
      const lines = indexes.map(index => selectedColumns.map(column => escapeValue(rows[index][column])).join(delimiter));
      return [header, ...lines].join("\\n");
    }}

    function splitTableColumns(partIndex) {{
      const resultColumns = splitColumns[partIndex] || [];
      return [...configColumns, ...resultColumns].filter((column, index, cols) => cols.indexOf(column) === index);
    }}

    async function copyText(text, copiedMessage) {{
      try {{
        await navigator.clipboard.writeText(text);
        status.textContent = copiedMessage;
      }} catch (error) {{
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
        status.textContent = copiedMessage;
      }}
    }}

    async function copyTsv(indexes, selectedColumns = currentColumns(), label = "Copied") {{
      const text = rowsToDelimited(indexes, selectedColumns, "\\t");
      await copyText(text, `${{label}} ${{indexes.length}} row(s) as TSV`);
    }}

    function downloadCsv(indexes, selectedColumns = currentColumns()) {{
      const text = rowsToDelimited(indexes, selectedColumns, ",");
      const blob = new Blob([text], {{type: "text/csv;charset=utf-8"}});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "fedmia_selected_rows.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
      status.textContent = `Downloaded ${{indexes.length}} selected row(s) as CSV`;
    }}

    function setActiveColumns(nextColumns) {{
      activeColumns.clear();
      nextColumns.filter(column => columns.includes(column)).forEach(column => activeColumns.add(column));
      render();
    }}

    function renderColumnChooser() {{
      const html = columns.map(column => {{
        const checked = activeColumns.has(column) ? "checked" : "";
        return `<label class="column-choice"><input type="checkbox" data-column="${{escapeText(column)}}" ${{checked}}> ${{escapeText(labels[column] || column)}}</label>`;
      }}).join("");
      if (columnChooser.innerHTML !== html) {{
        columnChooser.innerHTML = html;
        columnChooser.querySelectorAll("input[data-column]").forEach(box => {{
          box.addEventListener("change", event => {{
            const column = event.target.dataset.column;
            if (event.target.checked) activeColumns.add(column);
            else activeColumns.delete(column);
            render();
          }});
        }});
      }}
    }}

    Object.values(filters).forEach(element => element.addEventListener("input", render));
    document.getElementById("selectVisible").addEventListener("click", () => {{
      visibleIndexes().forEach(index => selected.add(index));
      render();
    }});
    document.getElementById("clearSelection").addEventListener("click", () => {{
      selected.clear();
      render();
    }});
    document.getElementById("copySelected").addEventListener("click", () => {{
      copyTsv(Array.from(selected), currentColumns(), "Copied selected");
    }});
    document.getElementById("copyVisible").addEventListener("click", () => {{
      copyTsv(visibleIndexes(), currentColumns(), "Copied visible");
    }});
    document.getElementById("downloadSelectedCsv").addEventListener("click", () => {{
      downloadCsv(Array.from(selected));
    }});
    document.getElementById("copySplitA").addEventListener("click", () => {{
      const indexes = selected.size ? Array.from(selected) : visibleIndexes();
      copyTsv(indexes, splitTableColumns(0), "Copied split A");
    }});
    document.getElementById("copySplitB").addEventListener("click", () => {{
      const indexes = selected.size ? Array.from(selected) : visibleIndexes();
      copyTsv(indexes, splitTableColumns(1), "Copied split B");
    }});
    document.getElementById("usePaperColumns").addEventListener("click", () => {{
      setActiveColumns(defaultColumns);
    }});
    document.getElementById("useAllColumns").addEventListener("click", () => {{
      setActiveColumns(columns);
    }});
    document.getElementById("useMainMetrics").addEventListener("click", () => {{
      setActiveColumns([...configColumns, ...splitColumns[0]]);
    }});
    document.getElementById("useLowFprMetrics").addEventListener("click", () => {{
      setActiveColumns([...configColumns, ...splitColumns[1]]);
    }});
    document.getElementById("clearColumns").addEventListener("click", () => {{
      setActiveColumns([]);
    }});

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    inputs = args.input or DEFAULT_INPUTS
    default_columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, columns = read_rows(inputs)
    static_columns = [column for column in default_columns if column in columns]
    html_path = output_dir / "fedmia_paper_table_view.html"
    csv_path = output_dir / "fedmia_paper_table_view.csv"
    tsv_path = output_dir / "fedmia_paper_table_view.tsv"

    write_delimited(csv_path, rows, static_columns, ",")
    write_delimited(tsv_path, rows, static_columns, "\t")
    html_path.write_text(make_html(rows, columns, static_columns), encoding="utf-8")

    print(f"Loaded {len(rows)} row(s).")
    print(f"Wrote {html_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {tsv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
