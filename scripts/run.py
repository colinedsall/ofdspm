""" run.py
Name:           run.py
Author:         Colin Edsall
Date:           June 6, 2025
Version:        3
Description:    Python script to run and train the ML model (RandomForestClassifier) with the given sample data in the sorted_data/
                directory.

                The user can select whether to show a sample image from the file selected using the --file argument in the CLI.
                The default file being predicted on is `sorted_data/good_images/Cs50MA50approx_0003.ibw`.

Changelog:      1:  (test.py) Original file. Does not have arguments nor CLI.
                2:  (run.py) Modified to contain some CLI and cleaned up comments.
                3:  Changed function name to use the single output, since the dictionary output version of the train func causes issues.
"""

from utils import train_ml_model_matched_single_out, enhanced_analysis, check_failure_flags, compute_thresholds, collect_all_stats, create_test_dataframe
from aespm import ibw_read
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing file for real image-based ML predictor.")
    parser.add_argument('--show', action='store_true', default=False, help='Show visualizations')
    parser.add_argument('--file', type=str, default="sorted_data/good_images/Cs50MA50approx_0003.ibw", help='Path to the IBW file to analyze')
    args = parser.parse_args()

    # Define the ML model from tools
    ml_model = train_ml_model_matched_single_out('sorted_data/bad_images', 'sorted_data/good_images', ibw_read)

    if args.show:
        subprocess.Popen([
            "python", "-c",
            (
                "import aespm.tools as at; "
                "import matplotlib.pyplot as plt; "
                f"at.display_ibw(r'{args.file}', key=['Height']); "
                "plt.show()"
            )
        ])

    # Choose the file from the arguments (defaulted)
    file = args.file

    # Create the test dataframe
    df = create_test_dataframe(file)

    # Used to create thresholds for checking (not necessary for ML, may need to change)
    folder = "sorted_data/good_images"
    stats_by_channel, pairwise_residuals, all_channels = collect_all_stats(folder)

    thresholds = compute_thresholds(stats_by_channel, pairwise_residuals)
    results = check_failure_flags(df, thresholds, True, ml_model)

    # Get analysis report
    enhanced_analysis(results)
