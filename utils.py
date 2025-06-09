""" utils.py
Name:           utils.py
Author:         Colin Edsall
Date:           June 6, 2025
Version:        1
Description:    This Python file contains many functions derived in the simulator notebook and referenced
                in scripts in this repo.
"""

from aespm import ibw_read
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from spmsimu.simulator import *
import os

"""
Train ML model using both good and bad image files directly.

Uses RandomForestsClassifier, not XGBoost for this example. Performance improvement not needed.
"""
def train_ml_model_matched(bad_image_files, good_image_files, ibw_read_func):

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    import pickle
    import os
    
    print("=== TRAINING ML MODEL (MATCHED TO PREDICTION) ===")
    
    # Use only the 4 channels that the prediction function uses
    target_channels = ['Height', 'Amplitude', 'Phase', 'ZSensor']
    
    print(f"Target channels: {target_channels}")
    
    # Calculate expected features: 4 stats per channel + pairwise residuals
    num_channels = len(target_channels)
    num_pairs = num_channels * (num_channels - 1) // 2
    expected_features = num_channels * 4 + num_pairs
    print(f"Expected features: {num_channels} channels * 4 stats + {num_pairs} pairs = {expected_features}")
    
    # Handle good image files
    if isinstance(good_image_files, str):
        if os.path.isdir(good_image_files):
            good_files = [os.path.join(good_image_files, f) for f in os.listdir(good_image_files) if f.endswith('.ibw')]
        else:
            good_files = [good_image_files]
    else:
        good_files = good_image_files
    
    # Handle bad image files
    if isinstance(bad_image_files, str):
        if os.path.isdir(bad_image_files):
            bad_files = [os.path.join(bad_image_files, f) for f in os.listdir(bad_image_files) if f.endswith('.ibw')]
        else:
            bad_files = [bad_image_files]
    else:
        bad_files = bad_image_files
    
    print(f"Found {len(good_files)} good image files")
    print(f"Found {len(bad_files)} bad image files")
    
    # Prepare feature matrix and labels
    X = []
    y = []
    
    # Feature names for consistency
    feature_names = []
    for ch_name in sorted(target_channels):
        feature_names.extend([f'{ch_name}_std', f'{ch_name}_range', f'{ch_name}_entropy', f'{ch_name}_skew'])
    
    # Add pairwise residual names
    sorted_channels = sorted(target_channels)
    for i in range(len(sorted_channels)):
        for j in range(i + 1, len(sorted_channels)):
            feature_names.append(f'{sorted_channels[i]}_{sorted_channels[j]}_residual')
    
    print(f"Feature names ({len(feature_names)}): {feature_names}")
    
    def extract_features_from_ibw(ibw_file, label):
        """Extract features from an IBW file"""
        try:
            ibw = ibw_read_func(ibw_file)
            
            # Create data dictionary like check_failure_flags expects
            data_dict = {}
            
            # Extract channel data based on ibw structure
            if hasattr(ibw.data, 'shape'):
                if len(ibw.data.shape) == 3:
                    # Assume channels are in the 3rd dimension
                    num_data_channels = ibw.data.shape[2]
                    # Map to your channel names (adjust this mapping as needed)
                    channel_mapping = ['Height', 'Amplitude', 'Phase', 'ZSensor'][:num_data_channels]
                    
                    for i, ch_name in enumerate(channel_mapping):
                        if ch_name in target_channels:
                            data_dict[ch_name] = ibw.data[:, :, i]
                            
                elif len(ibw.data.shape) == 2:
                    # Single channel or needs reshaping
                    if len(target_channels) == 1:
                        data_dict[target_channels[0]] = ibw.data
                    else:
                        # Try to split into channels
                        total_size = ibw.data.size
                        if total_size % len(target_channels) == 0:
                            pixels_per_channel = total_size // len(target_channels)
                            flattened = ibw.data.flatten()
                            for i, ch_name in enumerate(sorted(target_channels)):
                                start_idx = i * pixels_per_channel
                                end_idx = (i + 1) * pixels_per_channel
                                data_dict[ch_name] = flattened[start_idx:end_idx]
            
            # Extract features like check_failure_flags does
            features = []
            
            for ch_name in sorted(target_channels):
                if ch_name in data_dict:
                    channel_data = data_dict[ch_name].flatten()
                    
                    mean = np.mean(channel_data)
                    std = np.std(channel_data)
                    median = np.median(channel_data)
                    range_val = np.ptp(channel_data)
                    
                    # Compute entropy
                    hist, _ = np.histogram(channel_data, bins=256, density=True)
                    hist = hist[hist > 0]
                    entropy = -np.sum(hist * np.log(hist + 1e-12)) if len(hist) > 0 else 0
                    
                    skew = abs(mean - median)
                    features.extend([std, range_val, entropy, skew])
                else:
                    # Channel not available, use zeros
                    features.extend([0, 0, 0, 0])
            
            # Add pairwise residuals
            for i in range(len(sorted_channels)):
                for j in range(i + 1, len(sorted_channels)):
                    ch1_name, ch2_name = sorted_channels[i], sorted_channels[j]
                    if ch1_name in data_dict and ch2_name in data_dict:
                        ch1_data = data_dict[ch1_name].flatten()
                        ch2_data = data_dict[ch2_name].flatten()
                        
                        # Ensure same length
                        min_len = min(len(ch1_data), len(ch2_data))
                        residual = np.mean(np.abs(ch1_data[:min_len] - ch2_data[:min_len]))
                        features.append(residual)
                    else:
                        features.append(0)
            
            if len(features) == expected_features:
                X.append(features)
                y.append(label)
                return True
            else:
                print(f"Feature count mismatch for {os.path.basename(ibw_file)}: got {len(features)}, expected {expected_features}")
                return False
                
        except Exception as e:
            print(f"Failed to process {os.path.basename(ibw_file)}: {e}")
            return False
    
    # Process good images (label = 0)
    good_processed = 0
    for good_file in good_files:
        if extract_features_from_ibw(good_file, 0):  # 0 = good
            good_processed += 1
    
    print(f"Successfully processed {good_processed} good images")
    
    # Process bad images (label = 1)
    bad_processed = 0
    for bad_file in bad_files:
        if extract_features_from_ibw(bad_file, 1):  # 1 = bad
            bad_processed += 1
    
    print(f"Successfully processed {bad_processed} bad images")

    X = np.array(X)
    y = np.array(y)
    
    print(f"Final training data: {len(X)} samples, {X.shape[1]} features")
    print(f"Good images: {sum(y == 0)}, Bad images: {sum(y == 1)}")
    
    if len(X) == 0:
        print("No training data available!")
        return None
        
    if len(np.unique(y)) < 2:
        print("ERROR: Need both good and bad samples for training!")
        return None
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    print(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Create combined model
    combined_model = ScaledModel(model, scaler)
    
    with open('RandomForest_model.pkl', 'wb') as f:
        pickle.dump(combined_model, f)
    
    print("Model saved as 'RandomForest_model.pkl'")
    return {
        'model': combined_model,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train
    }

"""
Wrapper class to combine model and scaler
"""
class ScaledModel:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    @property
    def feature_importances_(self):
        return self.model.feature_importances_

# Load a trained ML model
def load_ml_model(model_path='image_quality_model.pkl'):
    import pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded ML model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"No ML model found at {model_path}. Train a model first or use use_ml=False")
        return None
    except Exception as e:
        print(f"Error loading ML model: {e}")
        return None

# Print detailed analysis of failure detection results
def enhanced_analysis(results):
    print("=== ENHANCED FAILURE ANALYSIS ===")
    
    print(f"Traditional failure: {results['traditional_failure']} (score: {results['traditional_failures_flagged']} of {results['traditional_failure_count']})")

    # Count entropy flags dynamically
    entropy_flag_count = sum(1 for key, value in results.items() 
                           if key.endswith('_entropy_flag') and value)
    total_entropy_flags = sum(1 for key in results.keys() 
                            if key.endswith('_entropy_flag'))
    
    print(f"Multiple entropy failure: {results['multiple_entropy_failure']} (score: {entropy_flag_count} flags of {total_entropy_flags})")
    
    print(f"High proximity failure: {results['high_proximity_failure']} (score: {results['proximity_score']:.3f})")
    print(f"ML failure: {results['ml_failure']} (probability: {results['ml_failure_probability']:.3f})")
    print(f"OVERALL FAILURE: {results['overall_failure']}")
    
    if 'proximity_details' in results:
        print("\n=== PROXIMITY TO THRESHOLDS ===")
        for ch, prox in results['proximity_details'].items():
            print(f"{ch}:")
            for metric, score in prox.items():
                status = "CLOSE" if score > 0.8 else "OK"
                print(f"  {metric}: {score:.3f} {status}")
    
    if 'feature_importance' in results:
        print("\n=== TOP CONTRIBUTING FEATURES (ML) ===")
        sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:8]:
            print(f"{feature}: {importance:.3f}")


# Checks the flags created after developing thresholds, not entirely necessary for ML training
def check_failure_flags(data_dict, thresholds, use_ml=True, ml_model=None):
    """
    Flexible version that works with any number of channels
    """
    results = {}
    
    # FILTER DATA FOR ML CONSISTENCY, only uses these four channels for consistency
    target_channels = ['Height', 'Amplitude', 'Phase', 'ZSensor']
    if use_ml and ml_model is not None:
        ml_data_dict = {ch: data for ch, data in data_dict.items() 
                        if ch in target_channels}
    else:
        ml_data_dict = data_dict
    
    def compute_features(channel):
        # Handle both numpy arrays and pandas Series
        if hasattr(channel, 'values'):
            channel_flat = channel.values.flatten()
        elif hasattr(channel, 'flatten'):
            channel_flat = channel.flatten()
        else:
            channel_flat = np.array(channel).flatten()
            
        mean = np.mean(channel_flat)
        std = np.std(channel_flat)
        median = np.median(channel_flat)
        range_val = np.ptp(channel_flat)
        hist, _ = np.histogram(channel_flat, bins=256, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-12))
        return mean, std, median, range_val, entropy
    
    # Extract features for all channels dynamically (use original data_dict for traditional analysis)
    channel_features = {}
    for ch_name, ch_data in data_dict.items():
        mean, std, median, range_val, entropy = compute_features(ch_data)
        channel_features[ch_name] = {
            'mean': mean, 'std': std, 'median': median, 
            'range': range_val, 'entropy': entropy
        }
    
    # Generate flags for all channels (use original logic)
    flag_count = 0
    entropy_flags = []
    
    for ch_name, features in channel_features.items():
        if ch_name in thresholds:
            # Individual channel flags
            results[f'{ch_name}_std_flag'] = features['std'] > thresholds[ch_name]['std_thresh']
            results[f'{ch_name}_range_flag'] = features['range'] > thresholds[ch_name]['range_thresh']
            results[f'{ch_name}_entropy_flag'] = features['entropy'] > thresholds[ch_name]['entropy_thresh']
            results[f'{ch_name}_skew_flag'] = abs(features['mean'] - features['median']) > thresholds[ch_name]['skew_thresh']
            
            # Count flags for this channel
            channel_flags = [
                results[f'{ch_name}_std_flag'],
                results[f'{ch_name}_range_flag'], 
                results[f'{ch_name}_entropy_flag'],
                results[f'{ch_name}_skew_flag']
            ]
            flag_count += sum(channel_flags)
            entropy_flags.append(results[f'{ch_name}_entropy_flag'])
    
    # Check pairwise residuals (use original data_dict)
    channel_names = list(data_dict.keys())
    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            ch1, ch2 = channel_names[i], channel_names[j]
            pair_name = f"{ch1}_{ch2}_residual"
            
            if pair_name in thresholds:
                # Convert to numpy arrays if needed
                ch1_data = data_dict[ch1].values if hasattr(data_dict[ch1], 'values') else data_dict[ch1]
                ch2_data = data_dict[ch2].values if hasattr(data_dict[ch2], 'values') else data_dict[ch2]
                
                residual = compute_pairwise_residual_mean(ch1_data.flatten(), ch2_data.flatten())
                results[f'{pair_name}_flag'] = residual > thresholds[pair_name]['threshold']
                flag_count += int(results[f'{pair_name}_flag'])
    
    # Overall failure assessment
    traditional_flags = [flag for key, flag in results.items() if key.endswith('_flag') and isinstance(flag, bool)]
    results['traditional_failure'] = any(traditional_flags)
    results['traditional_failure_count'] = len(traditional_flags)
    results['traditional_failures_flagged'] = sum(traditional_flags)
    
    # Enhanced rule-based logic
    results['multiple_entropy_failure'] = sum(entropy_flags) >= 2
    
    # Compute proximity scores (use original data_dict)
    proximity_scores = []
    proximity_details = {}
    
    for ch_name, features in channel_features.items():
        if ch_name in thresholds:
            std_prox = features['std'] / thresholds[ch_name]['std_thresh'] if thresholds[ch_name]['std_thresh'] > 0 else 0
            range_prox = features['range'] / thresholds[ch_name]['range_thresh'] if thresholds[ch_name]['range_thresh'] > 0 else 0
            skew_prox = abs(features['mean'] - features['median']) / thresholds[ch_name]['skew_thresh'] if thresholds[ch_name]['skew_thresh'] > 0 else 0
            
            proximity_scores.extend([std_prox, range_prox, skew_prox])
            proximity_details[ch_name] = {
                'std_proximity': std_prox,
                'range_proximity': range_prox,
                'skew_proximity': skew_prox
            }
    
    avg_proximity = np.mean(proximity_scores) if proximity_scores else 0
    results['high_proximity_failure'] = avg_proximity > 0.80
    results['proximity_score'] = avg_proximity
    results['proximity_details'] = proximity_details

    # ML-based prediction - FIXED FOR CONSISTENT FEATURE COUNT
    if use_ml and ml_model is not None:
        try:
            # Use only the filtered channels for ML prediction
            ml_channel_features = {}
            for ch_name, ch_data in ml_data_dict.items():
                mean, std, median, range_val, entropy = compute_features(ch_data)
                ml_channel_features[ch_name] = {
                    'mean': mean, 'std': std, 'median': median, 
                    'range': range_val, 'entropy': entropy
                }
            
            # Create feature vector dynamically from filtered channels
            feature_vector = []
            
            # Sort channel names for consistent ordering (same as training)
            sorted_channels = sorted(ml_channel_features.keys())
            
            # Add features for each channel (std, range, entropy, skew)
            for ch_name in sorted_channels:
                features = ml_channel_features[ch_name]
                feature_vector.extend([
                    features['std'],
                    features['range'], 
                    features['entropy'],
                    abs(features['mean'] - features['median'])  # skew
                ])
            
            # Add ALL pairwise residuals for the 4 channels (regardless of thresholds)
            # This ensures we always get the same 6 residuals that were used in training
            for i in range(len(sorted_channels)):
                for j in range(i + 1, len(sorted_channels)):
                    ch1, ch2 = sorted_channels[i], sorted_channels[j]
                    
                    # Compute residual regardless of whether it's in thresholds
                    ch1_data = ml_data_dict[ch1].values if hasattr(ml_data_dict[ch1], 'values') else ml_data_dict[ch1]
                    ch2_data = ml_data_dict[ch2].values if hasattr(ml_data_dict[ch2], 'values') else ml_data_dict[ch2]
                    residual = compute_pairwise_residual_mean(ch1_data.flatten(), ch2_data.flatten())
                    feature_vector.append(residual)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            print(f"ML feature vector shape: {feature_vector.shape}")  # Debug info
            print(f"Expected features: 4 channels * 4 stats + 6 residuals = 22")
            
            # Get ML prediction
            ml_prob = ml_model.predict_proba(feature_vector)[0][1]  # Probability of being bad
            results['ml_failure_probability'] = ml_prob
            results['ml_failure'] = ml_prob > 0.65
            
            # If it's a tree-based model, get feature importance for this prediction
            if hasattr(ml_model, 'feature_importances_'):
                # Generate feature names dynamically (same as training)
                feature_names = []
                for ch_name in sorted_channels:
                    feature_names.extend([
                        f'{ch_name}_std', f'{ch_name}_range', 
                        f'{ch_name}_entropy', f'{ch_name}_skew'
                    ])
                
                # Add residual feature names (all pairs)
                for i in range(len(sorted_channels)):
                    for j in range(i + 1, len(sorted_channels)):
                        ch1, ch2 = sorted_channels[i], sorted_channels[j]
                        feature_names.append(f'{ch1}_{ch2}_residual')
                
                results['feature_importance'] = dict(zip(feature_names, ml_model.feature_importances_))
                
        except Exception as e:
            print(f"ML prediction failed: {e}")
            results['ml_failure'] = False
            results['ml_failure_probability'] = 0.0
    else:
        results['ml_failure'] = False
        results['ml_failure_probability'] = 0.0
    
    # Count how many of the main failure modes are True (traditional, entropy, proximity, ML)
    sum_fails = sum([
        bool(results['traditional_failure']),
        bool(results['multiple_entropy_failure']),
        bool(results['high_proximity_failure']),
        bool(results['ml_failure'])
    ])

    results['overall_failure'] = sum_fails > 2
    
    # Store raw feature values for analysis - use original data_dict
    results['raw_features'] = {}
    for ch_name, features in channel_features.items():
        results['raw_features'][ch_name] = {
            'mean': features['mean'], 
            'std': features['std'], 
            'median': features['median'], 
            'range': features['range'], 
            'entropy': features['entropy']
        }
    
    # Add residual values to raw features (use original data_dict)
    channel_names = list(data_dict.keys())
    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            ch1, ch2 = channel_names[i], channel_names[j]
            pair_name = f"{ch1}_{ch2}_residual"
            
            if pair_name in thresholds:
                ch1_data = data_dict[ch1].values if hasattr(data_dict[ch1], 'values') else data_dict[ch1]
                ch2_data = data_dict[ch2].values if hasattr(data_dict[ch2], 'values') else data_dict[ch2]
                residual = compute_pairwise_residual_mean(ch1_data.flatten(), ch2_data.flatten())
                results['raw_features'][f'{pair_name}_mean'] = residual
    
    return results


# Computes channel statistics as needed from passed in values (array)
def compute_channel_stats(values):
    """Compute statistical features for a channel"""
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    value_range = np.ptp(values)
    
    # Compute entropy with better handling of edge cases
    hist, _ = np.histogram(values, bins=256, density=True)
    hist = hist[hist > 0]  # Remove zero bins to avoid log issues
    entropy = -np.sum(hist * np.log(hist + 1e-12))
    
    skew = abs(mean - median)  # Using absolute skew measure
    
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'range': value_range,
        'entropy': entropy,
        'skew': skew,
    }


def process_ibw_file(filepath):
    """Process a single IBW file and extract stats for all channels"""
    try:
        ibw = ibw_read(filepath)
        num_channels = ibw.data.shape[-1]
        reshaped = ibw.data.reshape(-1, num_channels)
        
        stats = {}
        for i, ch in enumerate(ibw.channels):
            stats[ch] = compute_channel_stats(reshaped[:, i])
        
        return stats, ibw.channels
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None


def compute_pairwise_residual_mean(data1, data2):
    """Compute mean absolute residual between two channels"""
    # Handle both numpy arrays and pandas Series
    if hasattr(data1, 'values'):
        data1 = data1.values
    if hasattr(data2, 'values'):
        data2 = data2.values
    return np.mean(np.abs(data1 - data2))


def collect_all_stats(folder):
    """Collect statistics from all IBW files in folder"""
    stats_by_channel = {}
    pairwise_residuals = {}
    all_channels = set()
    
    files = [f for f in os.listdir(folder) if f.endswith(".ibw")]
    
    for file in files:
        path = os.path.join(folder, file)
        stat_dict, channels = process_ibw_file(path)
        
        if stat_dict and channels:
            all_channels.update(channels)
            
            # Store channel statistics
            for channel, stats in stat_dict.items():
                stats['filename'] = file
                if channel not in stats_by_channel:
                    stats_by_channel[channel] = []
                stats_by_channel[channel].append(stats)
            
            # Compute pairwise residuals for all channel combinations
            try:
                ibw = ibw_read(path)
                num_channels = len(channels)
                reshaped = ibw.data.reshape(-1, num_channels)
                
                for i in range(num_channels):
                    for j in range(i + 1, num_channels):
                        pair_name = f"{channels[i]}_{channels[j]}_residual"
                        residual = compute_pairwise_residual_mean(reshaped[:, i], reshaped[:, j])
                        
                        if pair_name not in pairwise_residuals:
                            pairwise_residuals[pair_name] = []
                        pairwise_residuals[pair_name].append(residual)
                        
            except Exception as e:
                print(f"Error computing residuals for {file}: {e}")
    
    return stats_by_channel, pairwise_residuals, list(all_channels)


def compute_thresholds(stats_by_channel, pairwise_residuals, percentile=90):
    """Compute thresholds for all channels and pairwise residuals"""
    thresholds = {}
    
    # Compute thresholds for individual channels
    for ch, stats_list in stats_by_channel.items():
        df = pd.DataFrame(stats_list)
        thresholds[ch] = {
            'std_thresh': np.percentile(df['std'], percentile),
            'range_thresh': np.percentile(df['range'], percentile),
            'entropy_thresh': np.percentile(df['entropy'], 100 - percentile),  # Lower entropy = more uniform
            'skew_thresh': np.percentile(df['skew'], percentile),
        }
    
    # Compute thresholds for pairwise residuals
    for pair_name, residuals in pairwise_residuals.items():
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        thresholds[pair_name] = {
            'mean': residual_mean,
            'std': residual_std,
            'threshold': residual_mean + 3 * residual_std
        }
    
    return thresholds


def create_test_dataframe(file):
    """Create dataframe from IBW file with flexible channel support"""
    ibw = ibw_read(file)
    
    # Get dimensions directly from data shape
    if ibw.data.ndim == 3:
        side_y, side_x, channels = ibw.data.shape
    else:
        raise ValueError(f"Expected 3D data, got shape: {ibw.data.shape}")
    
    reshaped = ibw.data.reshape(-1, channels)
    rows, cols = np.meshgrid(range(side_y), range(side_x), indexing='ij')
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()
    
    # Create dataframe with dynamic columns based on available channels
    df_dict = {
        'row': rows_flat,
        'col': cols_flat,
    }
    
    # Add all channels dynamically
    for i, channel_name in enumerate(ibw.channels):
        df_dict[channel_name] = reshaped[:, i]
    
    df = pd.DataFrame(df_dict)
    
    print(f"Size of data (rows): {len(ibw.z)}")
    print(f"Current mode: {ibw.mode}")
    print(f"Channels: {ibw.channels}")
    print(f"Size (meters): {ibw.size}")
    
    return df