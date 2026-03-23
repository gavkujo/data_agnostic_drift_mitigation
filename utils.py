def print_metrics(metrics):
    print("\n=== Model Performance Metrics ===")
    print(f"AU-PRC on training set: {metrics['auprc_train']:.4f}")
    print(f"AU-PRC on test set after mitigation: {metrics['auprc_test']:.4f}")
