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
Train ML model using both good and bad image files directly.

Uses RandomForestsClassifier, not XGBoost for this example. Performance improvement not needed.
"""
def train_ml_model_matched_single_out(bad_image_files, good_image_files, ibw_read_func):

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
    return combined_model


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

""""
Synthetic image creation tools.
"""

def create_synthetic(height=256, width=256, channels=['Height', 'Amplitude', 'Phase', 'ZSensor'], 
                              severity='medium', seed=None, type='bad'):
    """
    Create synthetic images that replicate those from the sample set of the SFM. We can pass in the type
    as needed, but for now (by default), it generates bad images into dataframes for training the ML.
    
    Parameters:
    -----------
    height, width : int
        Image dimensions
    channels : list
        Channel names to generate
    severity : str
        'mild', 'medium', 'severe' - controls artifact intensity
    seed : int
        Random seed for reproducibility
    """

    from scipy import ndimage

    if seed is not None:
        np.random.seed(seed)
    
    # Severity parameters
    severity_params = {
        'mild': {'noise_scale': 0.5, 'line_prob': 0.025, 'gradient_strength': 0.5, 'scatter_density': 0.1},
        'medium': {'noise_scale': 1.5, 'line_prob': 0.05, 'gradient_strength': 1.0, 'scatter_density': 0.2},
        'severe': {'noise_scale': 2.5, 'line_prob': 0.075, 'gradient_strength': 2.0, 'scatter_density': 0.4}
    }
    params = severity_params[severity]
    
    def gradient_noise_2d(x, y):
        """Enhanced gradient noise function that works with 2D coordinates"""
        # Create position-dependent noise that follows curves/gradients
        r = np.sqrt(x**2 + y**2) + 1e-6  # Avoid division by zero
        angle = np.arctan2(y, x)
        
        # Multi-scale gradient noise
        noise1 = np.random.normal(np.log(r**0.01) + np.sqrt(r/100), params['noise_scale'])
        noise2 = np.random.normal(np.sin(angle * 3) * np.sqrt(r/50), params['noise_scale'] * 0.5)
        
        return noise1 + noise2
    
    def add_horizontal_lines(image, intensity_range=(-15, 45)):
        """Add horizontal line artifacts at random positions"""
        corrupted = image.copy()
        
        # Randomly select rows to corrupt
        num_lines = int(height * params['line_prob'])
        if num_lines > 0:
            corrupted_rows = np.random.choice(height, num_lines, replace=False)
            
            for row in corrupted_rows:
                # Random line characteristics
                line_intensity = np.random.uniform(*intensity_range)
                line_width = np.random.randint(2, 30) 
                
                # Add the line with some variation
                for w in range(line_width):
                    if row + w < height:
                        # Add some horizontal variation to make it look more realistic
                        variation = np.random.normal(0, 5, width)
                        corrupted[row + w, :] += line_intensity + variation
        
        return corrupted
    
    
    def add_scattered_noise(image, density=0.5):
        """Add scattered high-intensity noise points"""
        corrupted = image.copy()
        
        # Random scattered points
        num_points = int(height * width * density)
        if num_points > 0:
            rows = np.random.randint(0, height, num_points)
            cols = np.random.randint(0, width, num_points)
            
            # Random intensity spikes
            intensities = np.random.normal(0, params['noise_scale'] * 4, num_points)
            
            for r, c, intensity in zip(rows, cols, intensities):
                corrupted[r, c] += intensity
        
        return corrupted
    
    def add_gradient_drift(image):
        x = np.linspace(-1, 1, image.shape[1])
        y = np.linspace(-1, 1, image.shape[0])
        X, Y = np.meshgrid(x, y)

        angle = np.random.uniform(0, 2*np.pi)
        strength = params['gradient_strength'] * np.random.uniform(0.5, 2.0)

        gradient = strength * (np.cos(angle) * X + np.sin(angle) * Y) ** 3  # smoother paraboloid-like drift

        return image + gradient
    
    def add_stripe_artifacts(image):
        """Add periodic stripe artifacts (common in AFM)"""
        corrupted = image.copy()
        
        # Random stripe parameters
        if np.random.random() < 0.1: 
            stripe_period = np.random.randint(1, 5)
            stripe_direction = np.random.choice(['horizontal'])
            stripe_amplitude = params['noise_scale'] * 5
            
            if stripe_direction == 'horizontal':
                for i in range(height):
                    if i % stripe_period < stripe_period // 2:
                        corrupted[i, :] += stripe_amplitude * np.sin(np.linspace(0, 4*np.pi, width))
            else:
                for j in range(width):
                    if j % stripe_period < stripe_period // 2:
                        corrupted[:, j] += stripe_amplitude * np.sin(np.linspace(0, 4*np.pi, height))
        
        return corrupted
    
    def add_surface_points(image, num_points=30, min_sigma=1.5, max_sigma=4.0, amplitude_range=(0.5, 2.0)):
        """
        Add smooth Gaussian-like points (bumps or pits) to the image.
        
        Parameters:
        -----------
        image : np.ndarray
            The input image to modify.
        num_points : int
            Number of surface features to add.
        min_sigma, max_sigma : float
            Range for the Gaussian standard deviation (controls size of points).
        amplitude_range : tuple
            Range of amplitudes for the bumps (positive or negative).
        """
        h, w = image.shape
        new_image = image.copy()
        
        for _ in range(num_points):
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            sigma = np.random.uniform(min_sigma, max_sigma)
            amp = np.random.uniform(*amplitude_range)
            
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            
            gauss = amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            new_image += gauss
        
        return new_image

    def add_hard_surface_points(image, num_points=30, min_radius=2, max_radius=10, amplitude_range=(1.0, 5.0), shape='disk'):
        """
        Add sharp-edged surface features to the image: disks or cones.
        
        Parameters:
        -----------
        image : np.ndarray
            The input image to modify.
        num_points : int
            Number of features to add.
        min_radius, max_radius : int
            Radius range for the points.
        amplitude_range : tuple
            Height/depth range for the features.
        shape : str
            'disk' for flat-top bumps, 'cone' for linear sharp gradient
        """
        h, w = image.shape
        new_image = image.copy()

        for _ in range(num_points):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            radius = np.random.randint(min_radius, max_radius)
            amp = np.random.uniform(*amplitude_range)

            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            r2 = (X - cx)**2 + (Y - cy)**2

            if shape == 'disk':
                mask = r2 <= radius**2
                new_image[mask] += amp

            elif shape == 'cone':
                r = np.sqrt(r2)
                cone = amp * (1 - r / radius)
                cone[r > radius] = 0
                new_image += cone

        return new_image


    
    # Generate base images for each channel
    results = {}
    
    for channel in channels:
        # Start with a base pattern that varies by channel
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        X, Y = np.meshgrid(x, y)
        
        # Channel-specific base patterns
        if channel == 'Height':
            # Make a base image that mimics surface bumps or grains
            num_blobs = 100
            base = np.zeros((height, width))
            for _ in range(num_blobs):
                cx, cy = np.random.randint(0, width), np.random.randint(0, height)
                sigma = np.random.uniform(5, 25)
                blob = np.exp(-((X - cx/width*8 - 2)**2 + (Y - cy/height*8 - 2)**2) / (4 * (sigma/width*4)**2))
                base += blob
            base += 0.1 * np.random.randn(height, width)
        elif channel == 'Amplitude':
            base = np.exp(-(X**2 + Y**2)/4) + 0.2 * np.random.normal(0, 0.25, (height, width))
        elif channel == 'Phase':
            base = np.arctan2(Y, X) / np.pi + 0.2 * np.random.normal(0, 0.25, (height, width))
        elif channel == 'ZSensor':
            base = 0.3 * (X**2 + Y**2) + 0.25 * np.random.normal(0, 0.25, (height, width))
        else:
            # Generic pattern for other channels
            base = np.random.normal(0, 0.2, (height, width))
        
        # Apply gradient noise
        gradient_noise_map = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                gradient_noise_map[i, j] = gradient_noise_2d(X[i, j], Y[i, j])
        
        # Combine base pattern with gradient noise
        syn = base + gradient_noise_map * 1
        
        if type == 'bad':
            # Add various artifacts
            syn = add_horizontal_lines(syn)
            # syn = add_scattered_noise(syn, params['scatter_density'])
            # syn = add_stripe_artifacts(syn)
            syn = add_gradient_drift(syn)
            syn = add_surface_points(syn, num_points=250, min_sigma=5, max_sigma=75, amplitude_range=(-10, 10))
            syn = add_hard_surface_points(syn, num_points=750, min_radius=5, max_radius=30, amplitude_range=(5, 15), shape='disk')
            syn = add_hard_surface_points(syn, num_points=1000, min_radius=5, max_radius=15, amplitude_range=(-5, 25), shape='cone')

        else:
            # syn = add_gradient_drift(syn)
            syn = add_surface_points(syn, num_points=250, min_sigma=5, max_sigma=75, amplitude_range=(-10, 10))
            syn = add_hard_surface_points(syn, num_points=750, min_radius=5, max_radius=30, amplitude_range=(5, 15), shape='disk')
            syn = add_hard_surface_points(syn, num_points=1000, min_radius=5, max_radius=15, amplitude_range=(-5, 25), shape='cone')


        
        # Add some smoothing to make it look more realistic
        if np.random.random() < 0.25:
            sigma = np.random.uniform(0.5, 1)
            syn = ndimage.gaussian_filter(syn, sigma)
        
        results[channel] = syn
    
    return results

def synthetic_to_dataframe(image_dict, flatten=True):
    """Convert synthetic bad image dictionary to dataframe format"""
    height, width = next(iter(image_dict.values())).shape
    
    data = []
    for i in range(height):
        for j in range(width):
            row_data = {'row': i, 'col': j}
            for channel, image in image_dict.items():
                row_data[channel] = image[i, j]
            data.append(row_data)
    
    df = pd.DataFrame(data)
    
    if flatten:
        for channel in image_dict.keys():
            df[channel] = image_dict[channel].flatten()
    
    return df

def create_batch_synthetic_images(num_images=10, set_type='bad'):
    """Create a batch of synthetic bad images with varying characteristics"""
    batch = []
    severities = ['mild', 'medium', 'severe']
    
    for i in range(num_images):
        # Vary parameters for each image
        severity = np.random.choice(severities)
        seed = np.random.randint(0, 10000)
        
        # Randomly vary dimensions slightly
        h = np.random.randint(200, 300)
        w = np.random.randint(200, 300)
        
        image_dict = create_synthetic(
            height=h, width=w, 
            severity=severity, 
            seed=seed,
            type=set_type 
        )
        
        df = synthetic_to_dataframe(image_dict)
        batch.append({
            'dataframe': df,
            'image_dict': image_dict,
            'metadata': {'severity': severity, 'seed': seed, 'dimensions': (h, w)}
        })
    
    return batch

def visualize_synthetic_image(image_dict, title_prefix="Synthetic Generated"):
    num_channels = len(image_dict)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (channel, image) in enumerate(image_dict.items()):
        if idx < 4:  # Only show first 4 channels
            im = axes[idx].imshow(image, cmap='viridis', aspect='equal')
            axes[idx].set_title(f'{title_prefix} - {channel}')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], shrink=0.6)
    
    # Hide unused subplots
    for idx in range(len(image_dict), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

"""
Trace and retrace training and analysis tools.
"""

import os
import pickle
import numpy as np
from scipy.stats import pearsonr

# Set the folder path
sample_folder = os.path.join('scan_traces', 'MOBO')

# Get list of all .pickle files
files = sorted([
    os.path.join(sample_folder, f) for f in os.listdir(sample_folder)
    if f.endswith('.pickle')
])

# Helper function: Compute MAE and correlation between forward and backward traces
def compute_trace_metrics(trace_data):
    results = {}
    channel_names = ['Height', 'Amplitude', 'Phase', 'ZSensor']
    
    for i, name in enumerate(channel_names):
        fwd = trace_data[i * 2]
        bwd = trace_data[i * 2 + 1]
        
        mae = np.mean(np.abs(fwd - bwd))
        try:
            corr, _ = pearsonr(fwd, bwd)
        except:
            corr = np.nan
        
        results[name] = {'mae': mae, 'corr': corr}
    
    return results

def plot_traces(forward_avg, backward_avg, filename):
    n_channels = forward_avg.shape[0]
    chan_names = ['Height', 'Amplitude', 'Phase', 'ZSensor']

    fig, axs = plt.subplots(n_channels, 1, figsize=(12, 2.5 * n_channels), sharex=True)
    fig.suptitle(f"{filename}", fontsize=14)

    for c in range(n_channels):
        axs[c].plot(forward_avg[c], label='Forward', color='blue')
        axs[c].plot(backward_avg[c], label='Backward', color='orange', linestyle='--')
        axs[c].set_ylabel(chan_names[c])
        axs[c].legend()

    axs[-1].set_xlabel("Scan index")
    plt.tight_layout()
    plt.show()

from scipy.stats import entropy, skew, kurtosis
from scipy.integrate import simpson

def calculate_entropy(signal, bins=50):
    hist, bin_edges = np.histogram(signal, bins=bins, density=True)
    hist = hist + 1e-12  # avoid log(0)
    return entropy(hist)

def extract_features_from_file(pickle_path, show=False):
    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)

    traces = obj['traces']  # Shape: (N, M, C, L)
    x_measured = obj['x_measured']  # (N, D)
    param = obj['param']
    topo = obj['topo'] if 'topo' in obj else None

    if not np.isfinite(traces).all():
        print(f"Skipping {pickle_path}: contains NaN or Inf in traces")
        return []

    N, M, C, L = traces.shape

    forward_avg = np.mean(traces[:, :, 0::2, :], axis=(0, 1))  # (C/2, L)
    backward_avg = np.mean(traces[:, :, 1::2, :], axis=(0, 1))  # (C/2, L)
    if show: plot_traces(forward_avg, backward_avg, os.path.basename(pickle_path))

    features_list = []

    for n in range(N):
        # We'll average the M repeats for trace and retrace for stability
        forward_traces = traces[n, :, 0::2, :]  # M x (C/2) x L
        backward_traces = traces[n, :, 1::2, :] # M x (C/2) x L

        # Average over repeats
        forward_avg = np.mean(forward_traces, axis=0)  # (C/2, L)
        backward_avg = np.mean(backward_traces, axis=0)  # (C/2, L)

        n_channels = forward_avg.shape[0]

        # Features container for this sample
        features = {}

        # Trace vs retrace residual features per channel
        for c in range(n_channels):
            fwd = forward_avg[c]
            bwd = backward_avg[c]

            residual = fwd - bwd

            # Correlation
            corr = np.corrcoef(fwd, bwd)[0, 1]
            # MAE
            mae = np.mean(np.abs(residual))
            # Max abs error
            max_err = np.max(np.abs(residual))
            # Std of residual
            std_res = np.std(residual)
            # Integrated absolute difference (area)
            area_diff = simpson(np.abs(residual))

            # Basic trace stats on forward trace
            ent = calculate_entropy(fwd)
            sk = skew(fwd)
            kt = kurtosis(fwd)
            std_fwd = np.std(fwd)
            rng_fwd = np.ptp(fwd)

            # Store with channel name if known (assumed order: Height, Amplitude, Phase, ZSensor)
            chan_name = ['Height', 'Amplitude', 'Phase', 'ZSensor'][c]

            features[f'{chan_name}_corr'] = corr
            features[f'{chan_name}_mae'] = mae
            features[f'{chan_name}_max_err'] = max_err
            features[f'{chan_name}_std_residual'] = std_res
            features[f'{chan_name}_area_diff'] = area_diff
            features[f'{chan_name}_entropy'] = ent
            features[f'{chan_name}_skew'] = sk
            features[f'{chan_name}_kurtosis'] = kt
            features[f'{chan_name}_std_fwd'] = std_fwd
            features[f'{chan_name}_range_fwd'] = rng_fwd

        # Include controlling parameters for this sample (x_measured[n])
        if n < x_measured.shape[0]:
            for i, param_name in enumerate(['drive', 'setpoint', 'I_gain']):
                if i < x_measured.shape[1]:
                    features[param_name] = x_measured[n, i]
        else:
            # fallback, if x_measured missing
            for param_name in ['drive', 'setpoint', 'I_gain']:
                features[param_name] = np.nan

        # Inter-channel correlation residuals (example pairs)
        # Residuals between channels: difference of (trace - retrace) signals, then correlate
        pairs = [('Height','ZSensor'), ('Amplitude','Height'), ('Amplitude','Phase'), ('Phase','ZSensor')]
        for ch1, ch2 in pairs:
            idx1 = ['Height', 'Amplitude', 'Phase', 'ZSensor'].index(ch1)
            idx2 = ['Height', 'Amplitude', 'Phase', 'ZSensor'].index(ch2)

            res1 = forward_avg[idx1] - backward_avg[idx1]
            res2 = forward_avg[idx2] - backward_avg[idx2]
            corr_pair = np.corrcoef(res1, res2)[0,1]

            features[f'{ch1}_{ch2}_residual_corr'] = corr_pair

        # Optional: topo image stats (mean/std per channel)
        if topo is not None:
            for i, chan_name in enumerate(['Height', 'Amplitude', 'Phase', 'ZSensor']):
                if i < topo.data.shape[0]:
                    topo_chan = topo.data[i]
                    features[f'topo_{chan_name}_mean'] = np.mean(topo_chan)
                    features[f'topo_{chan_name}_std'] = np.std(topo_chan)

        features_list.append(features)

    return features_list

import glob

def classify_file(features_list, filename):
    # Average the features over all samples in this file
    df = pd.DataFrame(features_list)
    
    channels = ['Height', 'Amplitude', 'Phase', 'ZSensor']

    # Compute mean MAE and Corr per channel
    mean_mae = {ch: df[f'{ch}_mae'].mean() for ch in channels}
    mean_corr = {ch: df[f'{ch}_corr'].mean() for ch in channels}

    # Define thresholds
    thresholds = {
        'Height': {'mae_max': 3e-8, 'corr_min': 0.6, 'corr_sign': 1},
        'Amplitude': {'corr_min': 0.6, 'corr_sign': -1},
        'Phase': {'corr_min': 0.6, 'corr_sign': 1},
        'ZSensor': {'mae_max': 3e-8, 'corr_min': 0.6, 'corr_sign': 1}
    }

    # Check criteria
    pass_criteria = True

    # Height check
    if mean_mae['Height'] >= thresholds['Height']['mae_max'] or mean_corr['Height'] <= thresholds['Height']['corr_min']:
        pass_criteria = False

    # Amplitude check (only correlation)
    if not (mean_corr['Amplitude'] <= -thresholds['Amplitude']['corr_min']):
        pass_criteria = False

    # Phase check (correlation)
    if mean_corr['Phase'] <= thresholds['Phase']['corr_min']:
        pass_criteria = False

    # ZSensor check
    if mean_mae['ZSensor'] >= thresholds['ZSensor']['mae_max'] or mean_corr['ZSensor'] <= thresholds['ZSensor']['corr_min']:
        pass_criteria = False

    label = 'Good' if pass_criteria else 'Bad'

    # Print summary like your example
    print(f"=== File: {os.path.basename(filename)} ===")
    for ch in channels:
        print(f"{ch:<10} | MAE: {mean_mae[ch]:.3e} | Corr: {mean_corr[ch]:+.3f}")
    print(f"Classification: {label}")
    print()

    return label

# Main loop to go through pickle files
def process_all_pickles(folder_path):
    import pickle

    pickle_files = glob.glob(os.path.join(folder_path, '*.pickle'))
    results = {}

    for file_path in pickle_files:
        try:
            features = extract_features_from_file(file_path)

            if not features:  # Skip empty result (e.g., due to NaNs)
                continue

            # Ensure at least one valid feature set
            if all('Height_mae' not in fs for fs in features):
                print(f"Skipping {file_path}: missing expected keys")
                continue

            label = classify_file(features, file_path)
            results[os.path.basename(file_path)] = label

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results

def build_dataset(pickle_folder):
    import os
    files = [os.path.join(pickle_folder, f) for f in os.listdir(pickle_folder) if f.endswith('.pickle')]
    X = []
    y = []
    
    for file_path in files:
        try:
            features = extract_features_from_file(file_path)
            if not features:
                continue
            
            label = classify_file(features, file_path)  # returns 'Good' or 'Bad'
            
            # Aggregate features from all feature sets in this file (example: mean per feature)
            df = pd.DataFrame(features)
            mean_features = df.mean().to_dict()
            
            # Add features and label
            X.append(mean_features)
            y.append(1 if label == 'Good' else 0)  # encode Good=1, Bad=0
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def train_models_on_features(X, y, models_to_train='all'):
    """
    Train multiple ML models using extracted features from pickle files
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Feature matrix with samples as rows and features as columns
    y : list or numpy.array
        Labels (0 for bad, 1 for good)
    models_to_train : str or list
        'all' to train all models, or list of model names to train
        Available: ['RandomForest', 'XGBoost', 'SVM', 'LogisticRegression', 'GradientBoosting', 'ExtraTrees']
    
    Returns:
    --------
    dict : Dictionary containing trained models and evaluation results
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    from xgboost import XGBClassifier
    import pickle
    import os
    
    print("=== TRAINING MULTIPLE ML MODELS ON PICKLE FEATURES ===")
    
    # Define available models
    available_models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    # Determine which models to train
    if models_to_train == 'all':
        models_to_use = available_models
    else:
        models_to_use = {name: available_models[name] for name in models_to_train if name in available_models}
    
    print(f"Training models: {list(models_to_use.keys())}")
    
    # Convert to DataFrame if needed for easier processing
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    y = np.array(y)
    
    print(f"Training data: {len(X)} samples, {X.shape[1]} features")
    print(f"Good samples: {sum(y == 1)}, Bad samples: {sum(y == 0)}")
    
    if len(X) == 0:
        print("No training data available!")
        return None
        
    if len(np.unique(y)) < 2:
        print("ERROR: Need both good and bad samples for training!")
        return None

    # FIXED: Proper handling of mixed data types
    print("Processing mixed data types...")
    
    # Store original feature names
    original_feature_names = list(X.columns)
    
    # Separate numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Non-numeric features: {len(non_numeric_cols)}")
    
    # Start with numeric columns
    X_processed = X[numeric_cols].copy()
    
    # Handle non-numeric columns (encode categorical variables)
    label_encoders = {}
    for col in non_numeric_cols:
        print(f"Encoding categorical feature: {col}")
        le = LabelEncoder()
        # Handle NaN values in categorical columns
        X_cat = X[col].fillna('missing')
        X_processed[col] = le.fit_transform(X_cat.astype(str))
        label_encoders[col] = le
    
    # Now handle NaN values in numeric columns only
    if X_processed.select_dtypes(include=[np.number]).isnull().any().any():
        print("Warning: Found NaN values in numeric features. Filling with median values.")
        from sklearn.impute import SimpleImputer
        
        # Only impute numeric columns
        numeric_cols_final = X_processed.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        X_processed[numeric_cols_final] = imputer.fit_transform(X_processed[numeric_cols_final])
    
    # Convert to numpy array
    X_final = X_processed.values.astype(float)
    feature_names = list(X_processed.columns)
    
    print(f"Final processed features: {X_final.shape[1]}")
    print(f"Sample feature names: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")
    
    # Split data once for all models
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale data once
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train all models
    trained_models = {}
    results_summary = []
    
    print(f"\n{'='*60}")
    print("TRAINING AND EVALUATING MODELS")
    print(f"{'='*60}")
    
    for model_name, model in models_to_use.items():
        print(f"\n--- Training {model_name} ---")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Cross-validation with proper scoring
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            # Fallback if AUC fails
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        
        # Predictions for detailed evaluation
        y_pred = model.predict(X_test_scaled)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        print(f"Cross-validation score: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results_summary.append({
            'Model': model_name,
            'Train_Accuracy': train_score,
            'Test_Accuracy': test_score,
            'CV_Score_Mean': cv_mean,
            'CV_Score_Std': cv_std
        })
        
        # Create combined model with scaler and encoders
        combined_model = ScaledModelWithEncoders(model, scaler, label_encoders, feature_names, numeric_cols, non_numeric_cols)
        trained_models[model_name] = combined_model
        
        # Save individual model in trace_models folder
        filename = f'{model_name}_trace_model.pkl'
        os.makedirs("trace_models", exist_ok=True)
        model_path = os.path.join("trace_models", filename)
        with open(model_path, 'wb') as f:
            pickle.dump(combined_model, f)
        print(f"Model saved as '{model_path}'")
    
    # Results summary
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('CV_Score_Mean', ascending=False)
    
    print(f"\n{'='*60}")
    print("MODELS PERFORMANCE SUMMARY (Ranked by CV Score)")
    print(f"{'='*60}")
    print(results_df.to_string(index=False, float_format='%.3f'))
    
    # Feature importance for tree-based models
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top 15)")
    print(f"{'='*60}")
    
    for model_name, combined_model in trained_models.items():
        if hasattr(combined_model.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': combined_model.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} - Top 15 Features:")
            print(importance_df.head(15).to_string(index=False, float_format='%.4f'))
    
    # Save results summary
    results_path = os.path.join("trace_models", "training_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results_summary': results_df,
            'feature_names': feature_names,
            'original_feature_names': original_feature_names,
            'label_encoders': label_encoders,
            'numeric_cols': numeric_cols,
            'non_numeric_cols': non_numeric_cols,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        }, f)
    print(f"\nTraining results saved to '{results_path}'")
    
    return {
        'models': trained_models,
        'results_summary': results_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'label_encoders': label_encoders
    }


class ScaledModelWithEncoders:
    """
    Wrapper class that combines a trained model with its scaler and label encoders
    for easy prediction on new data with mixed types
    """
    def __init__(self, model, scaler, label_encoders, feature_names, numeric_cols, non_numeric_cols):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_names = feature_names
        self.numeric_cols = numeric_cols
        self.non_numeric_cols = non_numeric_cols
    
    def predict(self, X):
        """Predict on new data, handling encoding and scaling"""
        X_processed = self._preprocess(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities on new data"""
        X_processed = self._preprocess(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict_proba(X_scaled)
    
    def _preprocess(self, X):
        """Preprocess new data using the same transformations as training"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Start with numeric columns
        X_processed = X[self.numeric_cols].copy()
        
        # Apply label encoders to categorical columns
        for col in self.non_numeric_cols:
            if col in X.columns:
                # Handle unseen categories
                X_cat = X[col].fillna('missing').astype(str)
                
                # Transform known categories, assign -1 for unknown
                encoded_values = []
                for val in X_cat:
                    if val in self.label_encoders[col].classes_:
                        encoded_values.append(self.label_encoders[col].transform([val])[0])
                    else:
                        # Assign a default value for unseen categories
                        encoded_values.append(-1)
                
                X_processed[col] = encoded_values
        
        # Handle missing values
        if X_processed.isnull().any().any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                imputer.fit_transform(X_processed), 
                columns=X_processed.columns
            )
        
        return X_processed.values.astype(float)


class ScaledModel:
    """Wrapper class to combine model and scaler"""
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
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None


def build_dataset_from_pickles(pickle_folder, classify_function):
    """
    Build a dataset from pickle files using the extract_features_from_file function
    
    Parameters:
    -----------
    pickle_folder : str
        Path to folder containing pickle files
    classify_function : function
        Function that takes (features, file_path) and returns 'Good' or 'Bad'
    
    Returns:
    --------
    tuple : (X_df, y) where X_df is feature DataFrame and y is list of labels
    """
    import os
    import pandas as pd
    
    files = [os.path.join(pickle_folder, f) for f in os.listdir(pickle_folder) if f.endswith('.pickle')]
    
    X = []
    y = []
    failed_files = []
    
    print(f"Processing {len(files)} pickle files...")
    
    for i, file_path in enumerate(files):
        try:
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
            
            # Extract features using your function
            features_list = extract_features_from_file(file_path)
            
            if not features_list:
                print(f"  No features extracted from {os.path.basename(file_path)}")
                continue
            
            # Get classification for this file
            label = classify_function(features_list, file_path)  # returns 'Good' or 'Bad'
            
            # Aggregate features from all feature sets in this file (mean per feature)
            df_features = pd.DataFrame(features_list)
            mean_features = df_features.mean().to_dict()
            
            # Add features and label
            X.append(mean_features)
            y.append(1 if label == 'Good' else 0)  # encode Good=1, Bad=0
            
            print(f"  Successfully processed: {label} ({len(features_list)} feature sets)")
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            failed_files.append(file_path)
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for f in failed_files:
            print(f"  {os.path.basename(f)}")
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X)
    
    print(f"\nDataset built:")
    print(f"  Total samples: {len(X_df)}")
    print(f"  Good samples: {sum(y)}")
    print(f"  Bad samples: {len(y) - sum(y)}")
    print(f"  Features: {len(X_df.columns)}")
    
    return X_df, y


# Helper functions for easy use
def train_all_trace_models(X, y):
    """Train all available models on trace features"""
    return train_models_on_features(X, y, models_to_train='all')

def train_specific_trace_models(X, y, model_list):
    """Train specific models on trace features"""
    return train_models_on_features(X, y, models_to_train=model_list)

def load_trace_model(model_name):
    """Load a saved trace model"""
    model_path = os.path.join("trace_models", f'{model_name}_trace_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)
    
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns

def evaluate_model_auc_roc(model, X_test, y_test, model_name="Model", plot=True):
    """
    Evaluate a binary classification model using AUC-ROC metrics
    
    Parameters:
    -----------
    model : trained model object
        Must have predict_proba method
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
    model_name : str
        Name for the model (used in plots)
    plot : bool
        Whether to generate ROC curve plot
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    else:
        raise ValueError("Model must have predict_proba method for AUC-ROC calculation")
    
    # Get binary predictions
    y_pred = model.predict(X_test)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Alternative AUC calculation (should be the same)
    roc_auc_sklearn = roc_auc_score(y_test, y_proba)
    
    # Print results
    print(f"=== {model_name} AUC-ROC Evaluation ===")
    print(f"AUC-ROC Score: {roc_auc:.4f}")
    print(f"AUC-ROC (sklearn): {roc_auc_sklearn:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    if plot:
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{model_name} - ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'{model_name} - Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
    
    # Return metrics dictionary
    return {
        'auc_roc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def cross_validate_auc_roc(model, X, y, cv_folds=5, model_name="Model"):
    """
    Perform cross-validation with AUC-ROC scoring
    
    Parameters:
    -----------
    model : sklearn model object
        Untrained model
    X : array-like
        Features
    y : array-like
        Labels
    cv_folds : int
        Number of cross-validation folds
    model_name : str
        Name for the model
    
    Returns:
    --------
    dict : Cross-validation results
    """
    
    print(f"=== {model_name} Cross-Validation AUC-ROC ===")
    
    # Stratified K-Fold to maintain class balance
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross-validation with AUC scoring
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    
    print(f"Cross-Validation AUC-ROC Scores: {cv_scores}")
    print(f"Mean AUC-ROC: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    print(f"95% Confidence Interval: {cv_scores.mean():.4f} ± {1.96 * cv_scores.std():.4f}")
    
    return {
        'cv_scores': cv_scores,
        'mean_auc': cv_scores.mean(),
        'std_auc': cv_scores.std()
    }

def load_and_evaluate_pickled_model(model_path, X_test, y_test, model_name="Pickled Model"):
    """
    Load a pickled model and evaluate it with AUC-ROC
    
    Parameters:
    -----------
    model_path : str
        Path to the pickled model file
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str
        Name for the model
    
    Returns:
    --------
    dict : Evaluation results
    """
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Successfully loaded model from {model_path}")
        
        # Evaluate the model
        results = evaluate_model_auc_roc(model, X_test, y_test, model_name)
        
        return results
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def benchmark_multiple_models(models_dict, X_test, y_test, plot_comparison=True):
    """
    Compare multiple models using AUC-ROC
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model_name: model_object pairs
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    plot_comparison : bool
        Whether to create comparison plot
    
    Returns:
    --------
    dict : Results for all models
    """
    
    results = {}
    
    if plot_comparison:
        plt.figure(figsize=(10, 8))
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*50}")
        result = evaluate_model_auc_roc(model, X_test, y_test, model_name, plot=False)
        results[model_name] = result
        
        if plot_comparison:
            plt.plot(result['fpr'], result['tpr'], lw=2, 
                    label=f'{model_name} (AUC = {result["auc_roc"]:.4f})')
    
    if plot_comparison:
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Summary table
    print(f"\n{'='*50}")
    print("SUMMARY - AUC-ROC Comparison")
    print(f"{'='*50}")
    for model_name, result in results.items():
        print(f"{model_name:20s}: AUC = {result['auc_roc']:.4f}")
    
    return results

# Example usage functions
def example_with_trained_model(trained_model, X_test, y_test):
    return evaluate_model_auc_roc(trained_model, X_test, y_test, "RandomForest")

def example_with_pickled_model(X_test, y_test):
    return load_and_evaluate_pickled_model('RandomForest_model.pkl', X_test, y_test, "RandomForest")

def example_cross_validation(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    return cross_validate_auc_roc(model, X_train, y_train, cv_folds=5, model_name="RandomForest")

# Interpretation guide
def interpret_auc_roc_score(auc_score):
    """
    Provide interpretation of AUC-ROC score
    """
    print(f"\n=== AUC-ROC Score Interpretation ===")
    print(f"Your AUC-ROC Score: {auc_score:.4f}")
    
    if auc_score >= 0.9:
        interpretation = "Excellent"
    elif auc_score >= 0.8:
        interpretation = "Good"
    elif auc_score >= 0.7:
        interpretation = "Fair"
    elif auc_score >= 0.6:
        interpretation = "Poor"
    else:
        interpretation = "Very Poor (worse than random)"
    
    print(f"Interpretation: {interpretation}")
    print(f"""
    AUC-ROC Interpretation Guide:
    - 1.0: Perfect classifier
    - 0.9-1.0: Excellent
    - 0.8-0.9: Good  
    - 0.7-0.8: Fair
    - 0.6-0.7: Poor
    - 0.5-0.6: Very Poor
    - 0.5: Random classifier (no discriminative ability)
    - <0.5: Worse than random (but can be inverted)
    """)

def predict_file_class(pickle_path, model_path):
    """
    Predict classification for a pickle file using a trained model
    
    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file to classify
    model_path : str
        Path to the saved ScaledModel (.pkl file)
    
    Returns:
    --------
    tuple : (label, confidence) where label is 0/1 and confidence is the probability
    """
    
    # Load the ScaledModel
    with open(model_path, 'rb') as f:
        scaled_model = pickle.load(f)
    
    # Extract features from the file
    feature_dicts = extract_features_from_file(pickle_path)
    if not feature_dicts:
        raise ValueError("No features extracted from file.")
    
    # Convert to DataFrame and average across all entries (if multiple)
    df = pd.DataFrame(feature_dicts)
    mean_features = df.mean(numeric_only=True)
    
    print(f"Extracted {len(mean_features)} features from file")
    print(f"Available features: {list(mean_features.index)}")
    
    # Check if the underlying model has feature_names_in_ (scikit-learn 1.0+)
    underlying_model = scaled_model.model
    
    if hasattr(underlying_model, 'feature_names_in_'):
        expected_features = underlying_model.feature_names_in_
        print(f"Model expects {len(expected_features)} features: {list(expected_features)}")
        
        # Reindex to match expected feature order, fill missing with 0
        X = mean_features.reindex(expected_features).fillna(0.0).values.reshape(1, -1)
        
        # Check for missing features
        missing_features = set(expected_features) - set(mean_features.index)
        if missing_features:
            print(f"Warning: Missing features filled with 0.0: {missing_features}")
            
    elif hasattr(underlying_model, 'n_features_in_'):
        expected_n_features = underlying_model.n_features_in_
        print(f"Model expects {expected_n_features} features (no feature names available)")
        
        X = mean_features.values.reshape(1, -1)
        if X.shape[1] != expected_n_features:
            if X.shape[1] < expected_n_features:
                # Pad with zeros if we have fewer features
                padding = np.zeros((1, expected_n_features - X.shape[1]))
                X = np.concatenate([X, padding], axis=1)
                print(f"Warning: Padded {expected_n_features - mean_features.shape[0]} features with 0.0")
            else:
                # Truncate if we have more features
                X = X[:, :expected_n_features]
                print(f"Warning: Truncated to first {expected_n_features} features")
    else:
        # Fallback: use all available features
        print("Warning: Cannot determine expected feature count from model")
        X = mean_features.values.reshape(1, -1)
    
    print(f"Final feature vector shape: {X.shape}")
    
    # Use the ScaledModel's predict method (it handles scaling internally)
    label = scaled_model.predict(X)[0]
    
    # Get confidence if available
    confidence = None
    if hasattr(scaled_model, 'predict_proba'):
        try:
            proba = scaled_model.predict_proba(X)[0]
            confidence = proba.max()
        except:
            print("Warning: Could not get prediction probabilities")
    
    return int(label), confidence


def predict_file_class_verbose(pickle_path, model_path):
    """
    Verbose version that shows detailed information about features and predictions
    """
    try:
        label, confidence = predict_file_class(pickle_path, model_path)
        
        # Interpret results
        class_name = "Good" if label == 1 else "Bad"
        confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
        
        print(f"\n=== PREDICTION RESULTS ===")
        print(f"File: {pickle_path}")
        print(f"Prediction: {class_name} (label={label})")
        print(f"Confidence: {confidence_str}")
        
        return label, confidence
        
    except Exception as e:
        print(f"Error predicting file class: {e}")
        return None, None


def batch_predict_files(pickle_folder, model_path, file_pattern="*.pickle"):
    """
    Predict classifications for multiple files in a folder
    """
    import glob
    import os
    
    pattern = os.path.join(pickle_folder, file_pattern)
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return []
    
    results = []
    print(f"Processing {len(files)} files...")
    
    for i, file_path in enumerate(files):
        print(f"\n--- File {i+1}/{len(files)}: {os.path.basename(file_path)} ---")
        
        try:
            label, confidence = predict_file_class(file_path, model_path)
            class_name = "Good" if label == 1 else "Bad"
            
            result = {
                'file': os.path.basename(file_path),
                'path': file_path,
                'prediction': class_name,
                'label': label,
                'confidence': confidence
            }
            results.append(result)
            
            conf_str = f" (conf: {confidence:.3f})" if confidence else ""
            print(f"Result: {class_name}{conf_str}")
            
        except Exception as e:
            print(f"Error: {e}")
            result = {
                'file': os.path.basename(file_path),
                'path': file_path,
                'prediction': 'ERROR',
                'label': None,
                'confidence': None,
                'error': str(e)
            }
            results.append(result)
    
    # Summary
    successful = [r for r in results if r['prediction'] != 'ERROR']
    if successful:
        good_count = sum(1 for r in successful if r['label'] == 1)
        bad_count = len(successful) - good_count
        
        print(f"\n=== BATCH PREDICTION SUMMARY ===")
        print(f"Total files processed: {len(files)}")
        print(f"Successful predictions: {len(successful)}")
        print(f"Good predictions: {good_count}")
        print(f"Bad predictions: {bad_count}")
        print(f"Errors: {len(results) - len(successful)}")
    
    return results

def plot_trace_retrace(traces, scan_size_um, param_index=-1, repeat_index=-1, channel_names=None):
    """
    Plots forward (trace) and backward (retrace) scan lines for all channels.

    Parameters:
    - traces: np.ndarray, shape (N, M, C, L)
    - scan_size_um: float, total scan size in micrometers
    - param_index: int, index of which parameter set to show
    - repeat_index: int, index of which repeat to show
    - channel_names: list of str, optional names like ['Height', 'Amplitude', 'Phase', 'ZSensor']
    """
    if channel_names is None:
        channel_names = ['Channel {}'.format(i+1) for i in range(traces.shape[2] // 2)]

    x = np.linspace(0, scan_size_um, traces.shape[-1])
    n_channels = len(channel_names)

    fig, ax = plt.subplots(1, n_channels, figsize=(n_channels*4, 3))
    if n_channels == 1:
        ax = [ax]  # Make it iterable

    for i in range(n_channels):
        fwd = traces[param_index, repeat_index, i*2]
        bkd = traces[param_index, repeat_index, i*2 + 1]
        ax[i].plot(x, fwd, label='Trace', color='blue')
        ax[i].plot(x, bkd, label='Retrace', color='orange')
        ax[i].set_title(channel_names[i])
        ax[i].set_xlabel('Distance (μm)')
        ax[i].set_ylabel('Signal')
        ax[i].legend()

    plt.tight_layout()
    plt.show()

def view_topo_from_pickle(file_name):
    with open(file_name, 'rb') as fopen:
        obj = pickle.load(fopen)

    topo = obj['topo']
    titles = ['Height', 'Amplitude', 'Phase', 'ZSensor']
    fig, ax = plt.subplots(1, 4, figsize=[12, 2.5])
    for i in range(4):
        # Use topo.data[i] to access each channel directly
        im = ax[i].imshow(topo.data[i], origin='lower')
        ax[i].set_title(titles[i])
        plt.colorbar(im, ax=ax[i])

    plt.tight_layout()
    plt.show()

    """
    For finalized product, seek further documentation or splitting up this utility module.
    """