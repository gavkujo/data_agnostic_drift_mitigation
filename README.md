# Team <>

## NAISC 2026 | Adaptive Drift Intelligence Challenge

Per-feature drift typing with iterative self-training for robust churn prediction under distribution shift.

**Public dataset result: 0.893 AU-PRC** (baseline: 0.723, +23.5% relative improvement)

---

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python ./src/main.py --train_data_filepath <train_csv_path> --test_data_filepath <test_csv_path>
```

**Example with public data:**
```bash
python ./src/main.py --train_data_filepath data/train.csv --test_data_filepath data/test.csv
```

**Estimated runtime:** ~30 seconds on the public dataset (70K rows, 42 features). Scales to under 10 minutes on datasets up to 10M rows and 500 features.

---

### Outputs

The pipeline produces all required outputs automatically:

| Output | Location | Description |
|---|---|---|
| Drift summary table | Console | Per-feature drift type, description, and mitigation applied |
| Runtime | Console | Total execution time in seconds |
| AU-PRC metrics | Console | Train and test set AU-PRC |
| Predictions preview | Console | Head of predicted probabilities |
| `prediction.csv` | Root directory | CustomerID + probability_score for all test rows |
| `model.joblib` | Root directory | Trained LightGBM model |
| `dashboard_data.json` | Root directory | Data for the interactive dashboard |

---

### Dashboard

Start the dashboard server, then use the browser GUI to upload CSVs and run the pipeline:

```bash
python dashboard/server.py
```

Open **http://localhost:5050** — upload train/test CSVs, click **▶ Run Pipeline**, and results appear automatically. No terminal interaction needed.

The dashboard includes two tabs:
- **Overview**: metrics, drift severity chart, pipeline ablation, score distribution, drift summary table
- **All Predictions**: searchable, sortable table of all predictions with risk levels and CSV download

---

### Project Structure

```
.
├── src/                        # Solution source code
│   ├── main.py                 # Entry point — pipeline orchestration
│   ├── drift_detection.py      # Phase 1: per-feature drift diagnostics
│   ├── encoding.py             # Phase 2: feature encoding & mitigation
│   ├── adaptation.py           # Phase 3+4: temporal weighting, self-training
│   └── utils.py                # Shared constants, sampling, stats
├── dashboard/                  # Interactive web dashboard
│   ├── index.html              # Dashboard UI
│   └── server.py               # Lightweight HTTP server
├── prediction.csv              # Model predictions on public test set
├── model.joblib                # Trained model
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

### How It Works

1. **Diagnose**: Each feature is independently tested for distribution shift (KS/χ²), concept drift (temporal correlation stability), and format changes (new categories). Features are classified into drift types: none, covariate, concept, format, mixed, or severe.

2. **Mitigate**: Shifted numerics are quantile-mapped. All categoricals are target-encoded with case normalisation (resolves format changes implicitly). Severe mixed-drift features are dropped. Concept drift triggers temporal weighting.

3. **Adapt**: Iterative self-training progressively adapts the model to the test distribution using high-confidence pseudo-labels. Budget-constrained, safety-gated, with prediction averaging for stability.

---

### Requirements

- Python 3.10+
- LightGBM 4.6.0
- scikit-learn ≥ 1.3.0
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- joblib ≥ 1.3.0
