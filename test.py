from utils import train_ml_model_matched, enhanced_analysis, check_failure_flags, compute_thresholds, collect_all_stats, create_test_dataframe
from aespm import ibw_read

if __name__ == "__main__":
    ml_model = train_ml_model_matched('sorted_data/bad_images', 'sorted_data/good_images', ibw_read)

    file = "sorted_data/good_images/BSFO_AC_0007.ibw"
    try:
        df = create_test_dataframe(file)
        channels = ['Height', 'Amplitude', 'Phase', 'ZSensor']
        for i, ch in enumerate(channels):
            row, col = divmod(i, 2)
            at.display_ibw(file, [ch])  # Display IBW for each channel
    except:
        pass # Since some files don't have all the channels we test for

    folder = "sorted_data/good_images"
    stats_by_channel, pairwise_residuals, all_channels = collect_all_stats(folder)

    thresholds = compute_thresholds(stats_by_channel, pairwise_residuals)
    results = check_failure_flags(df, thresholds, True, ml_model)

    enhanced_analysis(results)

    """
    The model training needs more images.

    """