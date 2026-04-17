"""
Dashboard server — run this, open browser, upload CSVs, get results.
Usage: python src/serve_dashboard.py
Then open http://localhost:5050
"""
import os, sys, json, time, subprocess, tempfile, shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
import cgi

PORT = 5050
UPLOAD_DIR = tempfile.mkdtemp(prefix="naisc_")

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/dashboard/index.html"
        elif self.path == "/dashboard_data.json":
            self.path = "/dashboard_data.json"
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == "/run":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={"REQUEST_METHOD": "POST",
                             "CONTENT_TYPE": content_type})

                train_path = os.path.join(UPLOAD_DIR, "train.csv")
                test_path = os.path.join(UPLOAD_DIR, "test.csv")

                for name, dest in [("train", train_path), ("test", test_path)]:
                    if name in form:
                        item = form[name]
                        with open(dest, "wb") as f:
                            f.write(item.file.read())

                # Run pipeline
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                try:
                    t0 = time.time()
                    result = subprocess.run(
                        [sys.executable, "src/main.py",
                         "--train_data_filepath", train_path,
                         "--test_data_filepath", test_path],
                        capture_output=True, text=True, timeout=660,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    elapsed = time.time() - t0

                    # Read prediction.csv and build dashboard data from pipeline output
                    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    pred_path = os.path.join(proj_root, "prediction.csv")

                    dash_data = None
                    if os.path.exists(pred_path):
                        import csv
                        preds = []
                        with open(pred_path) as pf:
                            reader = csv.DictReader(pf)
                            for row in reader:
                                preds.append({"id": row.get("CustomerID",""), "score": round(float(row.get("probability_score",0)),4)})

                        # Parse stdout for metrics
                        stdout = result.stdout or ""
                        import re
                        auprc_match = re.findall(r'\|\s*(Train Set|Test Set)\s*\|\s*([\d.]+)', stdout)
                        auprc = {}
                        for label, val in auprc_match:
                            auprc[label.strip()] = float(val)

                        # Parse drift features from stdout
                        features = []
                        for line in stdout.split('\n'):
                            if '|' in line and 'drift' in line.lower() and 'Columns with' not in line and '---' not in line:
                                parts = [p.strip() for p in line.split('|') if p.strip()]
                                if len(parts) >= 4:
                                    features.append({"name": parts[0], "type": parts[1],
                                                     "drift_type": parts[2].split(" drift")[0] if "drift" in parts[2] else "unknown",
                                                     "effect": 0.0, "mitigation": parts[3]})
                                    # Try to extract effect from description
                                    ks_m = re.search(r'KS=([\d.]+)', parts[2])
                                    tvd_m = re.search(r'TVD=([\d.]+)', parts[2])
                                    if ks_m: features[-1]["effect"] = float(ks_m.group(1))
                                    elif tvd_m: features[-1]["effect"] = float(tvd_m.group(1))

                        n_feat_match = re.search(r'Features:\s*(\d+)', stdout)
                        n_shifted_match = re.search(r'Shifted:\s*(\d+)', stdout)
                        domain_match = re.search(r'Domain AUC:\s*([\d.]+)', stdout)
                        n_train_match = re.search(r'Train:\s*([\d,]+)', stdout)
                        n_test_match = re.search(r'Test:\s*([\d,]+)', stdout)

                        dash_data = {
                            "n_train": int(n_train_match.group(1).replace(',','')) if n_train_match else 0,
                            "n_test": int(n_test_match.group(1).replace(',','')) if n_test_match else 0,
                            "n_features": int(n_feat_match.group(1)) if n_feat_match else 0,
                            "n_shifted": int(n_shifted_match.group(1)) if n_shifted_match else 0,
                            "domain_auc": float(domain_match.group(1)) if domain_match else 0,
                            "auprc_train": auprc.get("Train Set", 0),
                            "auprc_test": auprc.get("Test Set", 0),
                            "runtime": round(elapsed, 1),
                            "features": features,
                            "unshifted_count": (int(n_feat_match.group(1)) if n_feat_match else 0) - len(features),
                            "predictions": preds,
                        }

                    response = {
                        "status": "success" if result.returncode == 0 else "error",
                        "stdout": result.stdout[-3000:] if result.stdout else "",
                        "stderr": result.stderr[-1000:] if result.stderr else "",
                        "elapsed": round(elapsed, 1),
                        "dashboard_data": dash_data,
                    }
                except subprocess.TimeoutExpired:
                    response = {"status": "timeout", "stdout": "", "stderr": "Pipeline exceeded 660s timeout"}
                except Exception as e:
                    response = {"status": "error", "stdout": "", "stderr": str(e)}

                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(400, "Expected multipart/form-data")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Quieter logging
        if "/run" in str(args):
            print(f"[DASHBOARD] {args[0]}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Dashboard server starting on http://localhost:{PORT}")
    print(f"Upload directory: {UPLOAD_DIR}")
    print("Open your browser and navigate to the URL above.")
    HTTPServer(("", PORT), DashboardHandler).serve_forever()
