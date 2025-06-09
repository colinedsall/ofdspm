""" trace_run.py
Name:           trace_run.py
Author:         Colin Edsall
Date:           June 9, 2025
Version:        1
Description:    Python script to run and train many ML models for given .pickle files which contain trace information.

                This script copies and expands upon the work done in the Jupyter notebook trace_retrace.ipynb.

Changelog:      Version 1:  No changes.
"""

import argparse
from utils import build_dataset_from_pickles, train_all_trace_models, benchmark_multiple_models, classify_file, predict_file_class, plot_trace_retrace
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing file for real image-based ML predictor.")
    parser.add_argument('--show', action='store_true', default=False, help='Show visualizations')
    parser.add_argument('--file', type=str, default="scan_traces/MOBO/250314_Droplet1_MOBO.pickle", help='Path to the IBW file to analyze')
    parser.add_argument('--predict', action='store_true', default=False, help='If called, predict from the given file in --file.')
    args = parser.parse_args()

    # Build dataset from pickle files
    X_df, y = build_dataset_from_pickles('scan_traces/MOBO', classify_file)

    # Train all models
    results = train_all_trace_models(X_df, y)

    # # Or train specific models
    # results = train_specific_trace_models(X_df, y, ['RandomForest', 'XGBoost'])

    # # Load a saved model later
    # model = load_trace_model('RandomForest')
    # predictions = model.predict(new_features)

    if results is not None:
        auc_results = benchmark_multiple_models(
            results['models'],     # Dictionary of all trained models
            results['X_test'],     # Test features
            results['y_test'],     # Test labels
            plot_comparison=args.show   # Creates comparison ROC plot
        )
    else:
        print("Model training failed: not enough data or only one class present in labels.")

    path = args.file
    label, confidence = predict_file_class(
        path,
        model_path="trace_models/RandomForest_trace_model.pkl"
    )
    print(f"Filepath predicted: {path}")
    print(f"Prediction: {'Good' if label == 1 else 'Bad'} (confidence: {confidence:.3f})")

    if args.show: 
        with open(path, 'rb') as fopen:
            obj = pickle.load(fopen)
        traces = obj['traces']

        plot_trace_retrace(traces, scan_size_um=obj['param']['ScanSize']*1e6,
                   param_index=-1, repeat_index=-1,
                   channel_names=['Height', 'Amplitude', 'Phase', 'ZSensor'])



