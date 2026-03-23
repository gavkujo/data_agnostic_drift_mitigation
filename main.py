import argparse
import time
from detection.detection_main import run_detection
from mitigation.mitigation_main import run_mitigation
from models import train_and_evaluate_model
from utils import print_metrics

def main():
    parser = argparse.ArgumentParser(description="NAISC2026 Drift Detection & Mitigation Pipeline")
    parser.add_argument('--train_data_filepath', type=str, required=True)
    parser.add_argument('--test_data_filepath', type=str, required=True)
    args = parser.parse_args()

    start_time = time.time()

    # Drift Detection
    drift_info = run_detection(args.train_data_filepath, args.test_data_filepath)

    # Drift Mitigation
    mitigated_train_path = run_mitigation(args.train_data_filepath, drift_info)

    drift_runtime = time.time() - start_time
    print(f"\n[INFO] Drift detection & mitigation runtime: {drift_runtime:.2f} seconds\n")

    # Model Training & Evaluation
    metrics = train_and_evaluate_model(mitigated_train_path, args.test_data_filepath)
    print_metrics(metrics)

if __name__ == "__main__":
    main()
