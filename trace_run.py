""" trace_run.py
Name:           trace_run.py
Author:         Colin Edsall
Date:           June 9, 2025
Version:        2
Description:    Python script to run and train many ML models for given .pickle files which contain trace information.

                This script copies and expands upon the work done in the Jupyter notebook trace_retrace.ipynb.

Changelog:      Version 1:  No changes.
                Version 2:  Changed to use subprocesses to show all images at once, instead of requiring the user to
                            manually go through each. This aids in batch processing.
"""

import argparse
import subprocess
import sys
import os
import tempfile
from utils import build_dataset_from_pickles, train_all_trace_models, benchmark_multiple_models, classify_file, predict_file_class, plot_trace_retrace, view_topo_from_pickle
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def save_and_display_plot(plot_func, *args, **kwargs):
    """
    Save a plot to a temporary file and display it in a subprocess
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_file.close()
    
    # Call the plotting function and save to temp file
    plot_func(*args, **kwargs)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Display the image in a subprocess
    display_image_subprocess(temp_file.name)
    
    return temp_file.name

def display_image_subprocess(image_path):
    """
    Display an image using a subprocess based on the operating system
    """
    try:
        if sys.platform.startswith('darwin'):  # macOS
            subprocess.Popen(['open', image_path])
        elif sys.platform.startswith('linux'):  # Linux
            subprocess.Popen(['xdg-open', image_path])
        elif sys.platform.startswith('win'):  # Windows
            subprocess.Popen(['start', image_path], shell=True)
        else:
            print(f"Unsupported platform: {sys.platform}")
            print(f"Please manually open: {image_path}")
    except Exception as e:
        print(f"Error opening image: {e}")
        print(f"Image saved to: {image_path}")

def modified_benchmark_multiple_models(models, X_test, y_test, plot_comparison=False):
    """
    Modified version of benchmark_multiple_models that uses subprocess for display
    """
    from utils import benchmark_multiple_models
    
    if plot_comparison:
        # Call the original function but capture the plot
        results = benchmark_multiple_models(models, X_test, y_test, plot_comparison=True)
        
        # If a plot was created, save and display it
        if plt.get_fignums():  # Check if any figures exist
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_file.close()
            plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
            plt.close('all')
            display_image_subprocess(temp_file.name)
        
        return results
    else:
        return benchmark_multiple_models(models, X_test, y_test, plot_comparison=False)

def modified_plot_trace_retrace(traces, scan_size_um, param_index=-1, repeat_index=-1, channel_names=None):
    """
    Modified version that saves and displays plot via subprocess
    """
    plot_trace_retrace(traces, scan_size_um=scan_size_um, 
                      param_index=param_index, repeat_index=repeat_index,
                      channel_names=channel_names)
    
    # Save and display the current figure
    if plt.get_fignums():
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close('all')
        display_image_subprocess(temp_file.name)

def modified_view_topo_from_pickle(pickle_path):
    """
    Modified version that saves and displays plot via subprocess
    """
    view_topo_from_pickle(pickle_path)
    
    # Save and display the current figure
    if plt.get_fignums():
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close('all')
        display_image_subprocess(temp_file.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing file for real image-based ML predictor.")
    parser.add_argument('--show', action='store_true', default=False, help='Show visualizations')
    parser.add_argument('--file', type=str, default="scan_traces/MOBO/250314_Droplet1_MOBO.pickle", help='Path to the IBW file to analyze')
    parser.add_argument('--predict', action='store_true', default=False, help='If called, predict from the given file in --file.')
    parser.add_argument('--keep-temp', action='store_true', default=False, help='Keep temporary image files (useful for debugging)')
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

    temp_files = []  # Keep track of temporary files for cleanup

    if results is not None:
        auc_results = modified_benchmark_multiple_models(
            results['models'], # Dictionary of all trained models
            results['X_test'], # Test features
            results['y_test'], # Test labels
            plot_comparison=args.show # Creates comparison ROC plot
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
        
        # Use modified plotting functions
        modified_plot_trace_retrace(traces, scan_size_um=obj['param']['ScanSize']*1e6,
                                   param_index=-1, repeat_index=-1,
                                   channel_names=['Height', 'Amplitude', 'Phase', 'ZSensor'])
        
        modified_view_topo_from_pickle(args.file)

    if not args.keep_temp:
        import atexit
        import glob
        
        def cleanup_temp_files():
            # Clean up any remaining temp files created by this script
            temp_pattern = os.path.join(tempfile.gettempdir(), 'tmp*.png')
            for temp_file in glob.glob(temp_pattern):
                try:
                    if os.path.getctime(temp_file) > (os.time.time() - 3600):  # Only clean recent files
                        os.unlink(temp_file)
                except:
                    pass
        
        atexit.register(cleanup_temp_files)