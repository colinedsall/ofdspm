"""
Filename:       train.py
Author:         Colin Edsall
Date:           June 5th, 2025
Version:        1
Description:    Combines the logic from the Jupyter notebook to create and train a XGBoostClassifier model to
                predict the classification of any given .ibw file.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import aespm.tools as at
from aespm import ibw_read

# Needed include for packaging
import pickle

""" Version: 1
Synthetic image creation: This logic works to create DataFrame objects that can be used to train the model.
It is important to note that this must be tuned to create images similar to those given in the sample,
else the model will have no idea what is "good" or "bad" other than what we tell it is.

Refer to the documentation below.
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
    
    def add_horizontal_lines(image, intensity_range=(-25, 25)):
        """Add horizontal line artifacts at random positions"""
        corrupted = image.copy()
        
        # Randomly select rows to corrupt
        num_lines = int(height * params['line_prob'])
        if num_lines > 0:
            corrupted_rows = np.random.choice(height, num_lines, replace=False)
            
            for row in corrupted_rows:
                # Random line characteristics
                line_intensity = np.random.uniform(*intensity_range)
                line_width = np.random.randint(5, 35) 
                
                # Add the line with some variation
                for w in range(line_width):
                    if row + w < height:
                        # Add some horizontal variation to make it look more realistic
                        variation = np.random.normal(0, 1, width)
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
            syn = add_scattered_noise(syn, params['scatter_density'])
            syn = add_gradient_drift(syn)
            syn = add_stripe_artifacts(syn)
            syn = add_surface_points(syn, num_points=250, min_sigma=5, max_sigma=75, amplitude_range=(-4, 4))
            syn = add_hard_surface_points(syn, num_points=1000, min_radius=10, max_radius=25, amplitude_range=(-5, 20), shape='disk')

        else:
            syn = add_gradient_drift(syn)
            syn = add_surface_points(syn, num_points=250, min_sigma=5, max_sigma=75, amplitude_range=(-4, 4))
            syn = add_hard_surface_points(syn, num_points=1000, min_radius=10, max_radius=25, amplitude_range=(-5, 20), shape='disk')


        
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

def visualize_synthetic_image(image_dict, title_prefix="Synthetic Bad"):
    """Visualize the generated bad images"""
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
ML training:    This is used to train (and store) a model that we can use later to predict and test.

The model chosen for speed and capability is XGBoostClassifier. Compare to RandomForestClassifier, it was more
accurate and capable of the large dataset.
"""
class ScaledModel:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_

"""
Train ML model using synthetic good and bad image sets with only the 4 channels
that check_failure_flags uses.

Inputs:
+ good_set (obj):   Good set of sample DataFrames (synthesized) for training
+ bad_set (obj):    Bad set of sample DataFrames (synthesized) for training
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train_ml_model_from_synthetic_sets(good_set, bad_set):
    print("=== TRAINING ML MODEL FROM SYNTHETIC SETS ===")

    target_channels = ['Height', 'Amplitude', 'Phase', 'ZSensor']
    available_channels = target_channels.copy()
    print(f"Using channels: {available_channels}")

    num_channels = len(available_channels)
    num_pairs = num_channels * (num_channels - 1) // 2
    expected_features = num_channels * 4 + num_pairs

    # Prepare features and labels
    X = []
    y = []

    feature_names = []
    for ch in sorted(available_channels):
        feature_names += [f"{ch}_std", f"{ch}_range", f"{ch}_entropy", f"{ch}_skew"]
    for i in range(len(available_channels)):
        for j in range(i + 1, len(available_channels)):
            feature_names.append(f"{available_channels[i]}_{available_channels[j]}_residual")

    def extract_features(image_dict):
        features = []
        for ch in sorted(available_channels):
            data = image_dict[ch].flatten()
            mean = np.mean(data)
            std = np.std(data)
            median = np.median(data)
            range_val = np.ptp(data)
            hist, _ = np.histogram(data, bins=256, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist + 1e-12)) if len(hist) > 0 else 0
            skew = abs(mean - median)
            features += [std, range_val, entropy, skew]
        for i in range(len(sorted(available_channels))):
            for j in range(i + 1, len(sorted(available_channels))):
                ch1 = sorted(available_channels)[i]
                ch2 = sorted(available_channels)[j]
                data1 = image_dict[ch1].flatten()
                data2 = image_dict[ch2].flatten()
                residual = np.mean(np.abs(data1 - data2))
                features.append(residual)
        return features

    for entry in good_set:
        try:
            feats = extract_features(entry['image_dict'])
            if len(feats) == expected_features:
                X.append(feats)
                y.append(0)
        except Exception as e:
            print(f"Failed to process good image: {e}")

    for entry in bad_set:
        try:
            feats = extract_features(entry['image_dict'])
            if len(feats) == expected_features:
                X.append(feats)
                y.append(1)
        except Exception as e:
            print(f"Failed to process bad image: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Final training data: {len(X)} samples, {X.shape[1]} features")
    print(f"Good images: {sum(y == 0)}, Bad images: {sum(y == 1)}")

    if len(X) == 0 or len(np.unique(y)) < 2:
        print("Not enough data or class diversity!")
        return None

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Estimate scale_pos_weight = (# negative samples) / (# positive samples)
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Note: XGBoost handles unscaled features well, but scaling is not harmful either.
    model.fit(X_train_scaled, y_train)

    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    combined_model = ScaledModel(model, scaler)

    with open('synthetic_image_quality_model.pkl', 'wb') as f:
        pickle.dump(combined_model, f)

    print("Model saved as 'synthetic_image_quality_model.pkl'")
    return combined_model

""" This function is mostly used to call both sets to be created.

Define the size variable for the amount of synthesized examples. Note, this scales poorly.
"""
def train_and_generate(size=10):
    bad_set = create_batch_synthetic_images(num_images=size, set_type='bad')
    good_set = create_batch_synthetic_images(num_images=size, set_type='good')

    model = train_ml_model_from_synthetic_sets(good_set, bad_set)
    return model




""" Sample (Real) Data Calculations
These are some helper functions to create statistics for basic testing of the images. No ML occurs,
yet they are helpful indicators for use while training the ML model to detect good and bad images.
"""

import os

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




""" Computing the Thresholds
This function works to compute the thresholds using the bounds created above (percentiles). These will be used
later to indicate whether the data is abnormally different from the sample set.
"""
import pprint

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

folder = "sample_data/good_images"
stats_by_channel, pairwise_residuals, all_channels = collect_all_stats(folder)

thresholds = compute_thresholds(stats_by_channel, pairwise_residuals)
pprint.pprint(thresholds)       # To view




""" Failure checking:
This function works to check and deliver data about the types of failures that may have occurred.

This function checks both traditional (statistical bounds created above), entropy, and high proximity to bounds, as well as the ML
created in this file.

Inputs:
+ data_dict (arr):      Dictionary of data to be checked. Typically this is in the form of a DataFrame from reading a .ibw file.
+ thresholds (arr):     Thresholds (arr) passed into from above functions. These define the bounds that must be checked for the 4 criterion.
+ use_ml (bool):        If True, then use ML in final decision calculations.
+ ml_model (ML):        The exact model (typically pickle retrieved) to be used in predictions.
"""
def check_failure_flags(data_dict, thresholds, use_ml=True, ml_model=None):
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

def enhanced_analysis(results):
    """Print detailed analysis of failure detection results - MADE DYNAMIC"""
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




"""
Main function call:

For CLI:
+ Use --show to show the synthetic sample, if needed
+ Use --generate to generate a new ML model. Configure the size of the synthesized set as needed.
+ Use --file {str} to choose the file other than the default to compare.

Actions:
+ Creates a synthetic image to view
+ Creates a batch of synthetic images to train a new model if requested
+ Checks the given file against created thresholds from given data and the ML's decision
+ Outputs results
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic SFM image analysis and ML training")
    parser.add_argument('--show', action='store_true', default=False, help='Show visualizations')
    parser.add_argument('--generate', action='store_false', default=False, help='Train and generate a ML model')
    parser.add_argument('--file', type=str, default="sample_data/bad_images/PositionC_30Cs70M0000.ibw", help='Path to the IBW file to analyze')
    args = parser.parse_args()

    # Create a single synthetic bad image
    bad_image = create_synthetic(severity='medium', seed=42, type='good')

    # Convert to dataframe
    df = synthetic_to_dataframe(bad_image)
    print(f"Generated dataframe shape: {df.shape}")
    print(f"Channels: {[col for col in df.columns if col not in ['row', 'col']]}")

    # Visualize
    if args.show:
        visualize_synthetic_image(bad_image)

    df = create_test_dataframe(args.file)

    # Only train and generate if you want to make a NEW model, else import
    if args.generate:
        # Create a batch
        batch = create_batch_synthetic_images(num_images=5)
        model = train_and_generate(size=1000)       # Set to 1000 here

    with open('synthetic_image_quality_model.pkl', 'rb') as f:
        model = pickle.load(f)

    results = check_failure_flags(df, thresholds, True, model)
    enhanced_analysis(results)
