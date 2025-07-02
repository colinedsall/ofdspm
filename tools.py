"""
Filename:           tools.py
Author:             Colin Edsall
Date:               June 26, 2025
Version:            2
Changelog:          (Version 1) Initial commit.
Description:        This python file contains scripts needed for training  either the Barlow Twins approach 
                    for a model to predict the condition of an AFM tip based on trace and image data, or the
                    hybrid model hypothesized.

                    The scripts in this file are from experimentation and may be configured to serve other 
                    purposes, such as various weights for the combined reward function and the number of 
                    epochs required for training.

                    Seek documentation in this repository (README.md) for the use cases of this code.

                    (Version 2) Comparison changes
                    Changed to allow plotting and comparison of a hybrid model to the Barlow
                    Twins model. We also begin looking at using both models in conjunction to receive
                    a certain prediction.
"""

# Torch, for accessing models
import torch                                        
import torchvision.models as models                

# Other imports for training
import os                                                           # Used for files/os
import torch.nn as nn                                               # CNN library
import torch.nn.functional as F
import torch.optim as optim                                         # Optimization library
from torch.utils.data import Dataset, DataLoader                    # Dataloader
import torchvision.models as models                                 # Backbone/base models.    
import torchvision.transforms as transforms                         # Image transforms
import numpy as np                                                  # Numpy
import pandas as pd                                                 # Pandas
from PIL import Image                                               # Image handling

# Statistics
from sklearn.metrics import accuracy_score, classification_report   # Classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay# Confusion matrices
from sklearn.metrics import roc_curve, auc, roc_auc_score           # ROC curves
from sklearn.preprocessing import label_binarize                    # Labeling
from scipy import stats                 
import random
import traceback
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Plotting/realtime data
import matplotlib.animation as animation
from collections import deque
from IPython.display import display, clear_output
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# PCA analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# API with SPM
from igor2 import binarywave as bw                                  # To read .ibw files
import aespm.tools as at                                            # AE-SPM library

# Errors and debug (some tools depreciated)
import traceback
import re
import json
from datetime import datetime
from itertools import cycle
import time

"""
Adaptation of the API to handle this experimental data case. Note: this does not impact other experiments 
that this method can handle, but the current library does not support the data needed for this kind of training.

Each given sample image contains 8 channels (4 with trace and retrace), of which we need to fix some functions
to work with their name scheme.
"""

def load_ibw(file, ss=False):
    '''
    Load the ibw file as an IBWData object.

    Input:
        file            - String: path to the ibw file
        ss              - Boolean: if True then the ibw file will be treated
                            as domain switching spectroscopy file.
    Output:
        IBWData object:
        self.z          - Numpy array: 2D numpy array containing topography channel. 
                         Default is the "Height" channel.
        self.size       - float: Map size in the unit of meter
        self.mode        - String: Imaging mode. Currently support "AC Mode", "Contact Mode", "PFM Mode"
                         "SS Mode" and "DART Mode".
        self.header     - Dict: All the setup information.
        self.channels   - list: List of channel names
        self.data       - Numpy array: An array of all the saved image data in this ibw file in
                         the same order as self.channels
    '''
    
    # Return an IBWData object
    return IBWData(file, ss=ss)

class IBWData(object):
    '''
    Data structure for AR IBW maps.

    Attributes:
        self.z          - Numpy array: 2D numpy array containing topography channel. 
                         Default is the "Height" channel.
        self.size       - float: Map size in the unit of meter
        self.mode        - String: Imaging mode. Currently support "AC Mode", "Contact Mode", "PFM Mode"
                         "Spec" and "DART Mode".
        self.header     - Dict: All the setup information.
        self.channels   - list: List of channel names
        self.data       - Numpy array: An array of all the saved image data in this ibw file in
                         the same order as self.channels
        
    Methods:
        None
    '''
    def __init__(self, path, ss=False):
        super(IBWData, self).__init__()

        self._load_ibw(path)

        # Spectroscopy files:
        if ss == True:
            self.mode = "Spec"
            try:
                self._load_ss()
            except IndexError:
                pass
        elif "ARDoIVCurve" in self.header:
            self.mode = "Spec"
            try:
                self._load_ss()
            except IndexError:
                pass
        # Image files:
        else:
            try:
                self.size = self.header['ScanSize']
                self.mode = self.header['ImagingMode']
                z_index = self.channels.index('Height')
                self.z = self.data[z_index]
                
                try:
                    # Separate DART mode from general PFM mode
                    if self.mode == "PFM Mode" or self.mode == "AC Mode":
                        if len(self.channels) > 4 and len(self.channels) < 8:
                            self.mode = "DART Mode"
                            self.channels = ['Height', 'Amplitude1', 'Amplitude2', 'Phase1', 'Phase2', 'Frequency']
                        elif len(self.channels) > 7:
                            self.channels = ['Height (T)', 'Height (RT)', 'Amplitude (T)', 'Amplitude (RT)',
                                            'Phase (T)', 'Phase (RT)', 'ZSensor (T)', 'ZSensor (RT)']
                        else:
                            self.channels = ['Height', 'Amplitude', 'Deflection', 'Phase']
                except IndexError:
                    pass
            except KeyError:
                pass

    def _load_ibw(self, path):

        t = bw.load(path)
        wave = t.get('wave')

        # Decode the notes section to parse the header
        if isinstance(wave['note'], bytes):
            try:
                parsed_string = wave['note'].decode('utf-8').split('\r')
            except:
                parsed_string = wave['note'].decode('ISO-8859-1').split('\r')

        # Load the header
        self.header = {}

        for item in parsed_string:
            try:
                key, value = item.split(':', 1)
                value = value.strip()  # Remove leading/trailing whitespace
            except ValueError:
                continue  # For items that do not split correctly

            # Determine the data type of the value and convert
            if '.' in value or 'e' in value:  # Floating point check
                try:
                    self.header[key] = float(value)
                except ValueError:
                    self.header[key] = value
            elif value.lstrip('-').isdigit():  # Integer check
                self.header[key] = int(value)
            else:
                self.header[key] = value

        # Load the data
        # Transpose the data matrix
        self.data = wave['wData'].T
        # data = wave['wData']

        # Load the channel names
        self.channels = [self.header.get(f'Channel{i+1}DataType', 'Unknown') for i in range(np.shape(self.data)[0])]

    def _load_ss(self, nan=True, drop=0.1):

        if nan is True:
            bias_raw= self.data[-1]
            index_not_nan = np.where(~np.isnan(bias_raw))

            bias = bias_raw[index_not_nan]
            amp1, amp2, phase1, phase2, freq = self.data[1][index_not_nan], self.data[2][index_not_nan], self.data[3][index_not_nan], \
                        self.data[4][index_not_nan], self.data[5][index_not_nan]
        else:
            bias = self.data[-1]
            amp1 = self.data[1]
            amp2 = self.data[2]
            phase1 = self.data[3]
            phase2 = self.data[4]
            freq = self.data[5]

        #Correcting first without offset to calculate the drive parameters
        phase1 = self._correct_phase_wrapping(phase1, offset_correction=False)
        phase2 = self._correct_phase_wrapping(phase2, offset_correction=False)

        df = self.header['DFRTFrequencyWidth']

        a_dr, ph_dr, q = self._calc_drive_params(amp1, amp2, phase1/180*np.pi, phase2/180*np.pi, freq, df)

        phase1 = self._correct_phase_wrapping(phase1)
        phase2 = self._correct_phase_wrapping(phase2)
        ph_dr  = self._correct_phase_wrapping(ph_dr/np.pi*180)

        # Let's count how many times the bias has changed (on->off, off->on)
        index_bp = np.where(np.diff(bias) != 0)[0] + 1
        # We define the width of applied voltage by the first non-zero voltage plateau
        index_delta = index_bp[1] - index_bp[0]
        # We drop the first segment of zero bias signals as this is the initial settling time
        index_bp = np.concatenate([[index_bp[0]-index_delta], index_bp])

        # Output array length
        length = len(index_bp) // 2

        bias_on, bias_off = np.zeros(length), np.zeros(length)

        phase1_on,phase1_off = np.zeros(length), np.zeros(length)
        phase2_on, phase2_off  = np.zeros(length), np.zeros(length)
        amp_on, amp_off = np.zeros(length), np.zeros(length)
        amp1_on, amp1_off = np.zeros(length), np.zeros(length)
        amp2_on, amp2_off = np.zeros(length), np.zeros(length)
        freq_on, freq_off = np.zeros(length), np.zeros(length)

        amp_dr_on, amp_dr_off = np.zeros(length), np.zeros(length)
        phase_dr_on, phase_dr_off = np.zeros(length), np.zeros(length)
        q_on, q_off = np.zeros(length), np.zeros(length)

        # We drop the first and last 10% data to avoid oscillation after bias change (10% settling time)
        skip = int(drop * index_delta)
        
        for i in range(length * 2-1):
            start = index_bp[i] + skip
            end = index_bp[i+1] - skip
            if i % 2 == 0: # bias off
                phase1_off[i//2] = np.mean(phase1[start:end])
                phase2_off[i//2] = np.mean(phase2[start:end])
                amp1_off[i//2] = np.mean(amp1[start:end])
                amp2_off[i//2] = np.mean(amp2[start:end])
                freq_off[i//2] = np.mean(freq[start:end])
                bias_off[i//2] = np.mean(bias[start:end])

                phase_dr_off[i // 2] = np.mean(ph_dr[start:end])
                q_off[i // 2] = np.mean(q[start:end])
                amp_dr_off[i // 2] = np.mean(a_dr[start:end])

            else:
                bias_on[i//2] = np.mean(bias[start:end])
                phase1_on[i//2] = np.mean(phase1[start:end])
                phase2_on[i//2] = np.mean(phase2[start:end])
                amp1_on[i//2] = np.mean(amp1[start:end])
                amp2_on[i//2] = np.mean(amp2[start:end])
                freq_on[i//2] = np.mean(freq[start:end])

                phase_dr_on[i // 2] = np.mean(ph_dr[start:end])
                q_on[i // 2] = np.mean(q[start:end])
                amp_dr_on[i // 2] = np.mean(a_dr[start:end])

        self.bias = bias_on
        self.phase1_on = phase1_on
        self.phase1_off = phase1_off
        self.phase2_on = phase2_on
        self.phase2_off = phase2_off
        self.freq_on = freq_on
        self.freq_off = freq_off
        self.amp_on = amp1_on
        self.amp_off = amp1_off
        self.amp1_on = amp1_on
        self.amp1_off = amp1_off
        self.amp2_on = amp2_on
        self.amp2_off = amp2_off
        self.x_on = amp_on * np.cos(phase1_on/180*np.pi)
        self.x_off = amp_off * np.cos(phase1_off / 180 * np.pi)

        self.amp_dr_on = amp_dr_on
        self.amp_dr_off = amp_dr_off
        self.x_dr_on = amp_dr_on * np.cos(phase_dr_on / 180 * np.pi)
        self.x_dr_off = amp_dr_off * np.cos(phase_dr_off / 180 * np.pi)
        self.phase_dr_on = phase_dr_on
        self.phase_dr_off = phase_dr_off
        self.q_on = q_on
        self.q_off = q_off

        # return bias[1:], amp_off[1:], phase1_off[1:], phase2_off[1:]

    def _correct_phase_wrapping(self, ph, lower=-90, upper=270, offset_correction=True):
        '''
        Correct the phase wrapping in Jupiter.
        
        Input:
            Ph     - Array: array of phase values
            lower - float: lower bound of phase limit in your instrument
            upper - float: upper bound of phase limit in your instrument
        Output:
            ph_shift - Array: phase with wrapping corrected
        '''
        # Use the phase value measured at last pixel as the offset in the lock-in
        if offset_correction:
            ph_shift = ph - ph[-1]
        else:
            ph_shift = ph

        index_upper = np.where(ph_shift > upper)
        index_lower = np.where(ph_shift < lower)
        ph_shift[index_upper] -= 360
        ph_shift[index_lower] += 360

        return ph_shift

    @staticmethod
    def _calc_drive_params(_a1, _a2, _ph1, _ph2, _fc, _df):
        '''
        Calculate real Dart parameters from the observables.

        Input:
            _a1  - amplitude 1
            _a2  - amplitude 2
            _ph1 - phase 1
            _ph2 - phase 2
            _fc  - resonance frequency
            _df  - difference between freq 2 and freq 1
        Output:
            _a_drive  - drive amplitude
            _ph_drive - resonance phase
            _q        - resonanse quality factor
        '''

        epsilon = 1e-10  # a small adding for calculation stability
        _dph = _ph2 - _ph1
        _f1 = _fc - _df / 2
        _f2 = _fc + _df / 2

        _om = _f1 * _a1 / (_f2 * _a2)
        _fi = np.tan(_dph)

        _x1 = -(1 - np.sign(_fi) * np.sqrt(1 + np.square(_fi)) / _om) / (_fi + epsilon)
        _x2 = (1 - np.sign(_fi) * np.sqrt(1 + np.square(_fi)) * _om) / (_fi + epsilon)

        _q = np.sqrt(_f1 * _f2 * (_f2 * _x1 - _f1 * _x2) * (_f1 * _x1 - _f2 * _x2)) / (np.square(_f2) - np.square(_f1))
        _q[_q > 1000] = 1000
        _a_drive = _a1 * np.sqrt((_fc**2 - _f1**2)**2 +(_fc * _f1 / _q)**2) / np.square(_fc)
        _ph_drive = _ph1 - np.arctan(_fc * _f1 / (_q * (np.square(_fc) - np.square(_f1))))

        return _a_drive, _ph_drive, _q

def find_channel(obj, key):
    if key is None:
        return np.arange(len(obj.channels))
    else:
        index = []
        channels = obj.channels

        for item in key:
            if item in channels:
                index.append(channels.index(item))
        return index

def display_ibw(file, key=None, titles=None, display_index=None, cmaps=None, save=None, **kwargs):
    '''
    Display a single ibw with specified by the file path.

    Input:
        file    - Required: path to the file to be displayed or the loaded ibw object returned by load_ibw()
        key     - Optional: list of channels to be displayed
        titles     - Optional: list of titles corresponding to the key
        display_index - Optional: index provided by display_ibw_folder() function
        cmaps     - Optional: list of color maps that will be used for each channels in key
        save     - Optional: if None, no image will be saved. If not None, each image will be 
                            saved as fileName + save
        **kwarg - Optional: Additional keyword arguments are sent to imshow().

    Output:
        ibw_files   -list: ibw file names in the same order as they are displayed
        data        -list: SciFiReader object of each ibw file displayed

    Example use:
        display_ibw(file_path, 
                titles=['Height (T)', 'Amplitude (T)', 'Phase (T)', 'ZSensor (T)'],
                key=['Height (T)', 'Amplitude (T)', 'Phase (T)', 'ZSensor (T)'])
    '''

    try:
        if type(file) is str:
            t = load_ibw(file)
        else:
            t = file

        if key is not None:
            if not isinstance(key, list):
                key = list(key)
        else:
            key = t.channels

        if titles is not None:
            if not isinstance(titles, list):
                titles = list(titles)

        if cmaps is not None:
            if not isinstance(cmaps, list):
                cmaps = list(cmaps)

        if t.mode != 'Spec': # skip the spectrum ibw files
            indices = find_channel(obj=t, key=key)
            if len(indices) == 1: # Only one channel will be displayed
                plt.figure(figsize=[4,4])
                to_plot = t.data[indices[0]]
                if cmaps is None:
                    im = plt.imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], **kwargs)
                else:
                    im = plt.imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], cmap=cmaps[0], **kwargs)
                if display_index is None:
                    title = "{}: {}".format(t.mode, key[0]) if not titles else titles
                else:
                    title = "{}: {}-{}".format(display_index, t.mode, key[0]) if not titles else titles

                plt.title(title)
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.tight_layout()
                if save is not None:
                    plt.savefig('{}.png'.format(save), dpi=400, bbox_inches='tight', pad_inches=0.1)
            else:
                n_cols = len(indices)
                fig,ax=plt.subplots(1, n_cols, figsize=[n_cols*3+1, 3])
                for i in range(len(indices)):
                    to_plot = t.data[indices[i]]
                    if cmaps is None:
                        im = ax[i].imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], **kwargs)
                    else:
                        im = ax[i].imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], cmap=cmaps[i], **kwargs)
                    divider = make_axes_locatable(ax[i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)
                    if titles is None:
                        if not i:
                            ax[i].set_title("{}: {}-{}".format(display_index, t.mode, t.channels[indices[i]]))
                        else:
                            ax[i].set_title("{}".format(t.channels[indices[i]]))
                    else:
                        ax[i].set_title(titles[i])
                    plt.tight_layout()
                    if save is not None:
                        plt.savefig('{}.png'.format(save), dpi=400, bbox_inches='tight', pad_inches=0.1)
        else:
            pass
    except TypeError:
        pass

def summarize_ibw_file(filename):
    """
    Prints a summary of information available in a .ibw file.

    Input:
        filename            - Filepath to .ibw file
    
    Output:
        IBWData object:
        self.z          - Numpy array: 2D numpy array containing topography channel. 
                         Default is the "Height" channel.
        self.size       - float: Map size in the unit of meter
        self.mode        - String: Imaging mode. Currently support "AC Mode", "Contact Mode", "PFM Mode"
                         "SS Mode" and "DART Mode".
        self.header     - Dict: All the setup information.
        self.channels   - list: List of channel names
        self.data       - Numpy array: An array of all the saved image data in this ibw file in
                         the same order as self.channels

    Example:
        file_path = 'exp_data/read_out/Read_out_0035.ibw'
        summarize_ibw_file(file_path)
    """

    ibw = load_ibw(filename)

    print("File :", filename)
    print("Imaging Mode :", ibw.mode)
    print("Scan Size :", ibw.size)
    print("Header keys :", list(ibw.header.keys()))

    # Header keys are not defined for this sample, since they only go to four channels
    # Instead, we *know* the label style, so we can just give the channels each name and capture
    # the data that way.

    print("Available Channels :", ibw.channels)
    print("Data Shape :", ibw.data.shape)

    df = pd.DataFrame(ibw.data[0])
    df.to_csv('ibw_dataframe.csv', index=False)

    return ibw

def is_jupyter():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False

def clear_console():
# For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

"""
Hybrid model training functions and class definitions for adaptive loss training and a reward-aware model.
"""

class AdaptiveLoss(nn.Module):
    def __init__(self):
        super(AdaptiveLoss, self).__init__()
        # Learnable weights for multi-task learning
        self.log_var_class = nn.Parameter(torch.zeros(1))
        self.log_var_reward = nn.Parameter(torch.zeros(1))

    def forward(self, class_pred, class_true, reward_pred, reward_true):
        # Classification loss
        class_loss = F.cross_entropy(class_pred, class_true)

        # Reward loss (MSE)
        reward_loss = F.mse_loss(reward_pred.squeeze(), reward_true)

        # Adaptive weighting based on uncertainty
        precision_class = torch.exp(-self.log_var_class)
        precision_reward = torch.exp(-self.log_var_reward)

        # Total loss with automatic balancing
        total_loss = (precision_class * class_loss + self.log_var_class + 
                     precision_reward * reward_loss + self.log_var_reward)

        return total_loss, class_loss, reward_loss

def load_multi_dataset(base_folders, dataset_names):
    """
    Load IBW files from multiple datasets and maintain proper scan indices
    
    Inputs:
        base_folders: list of base folder paths
        dataset_names: list of dataset subfolder names
    
    Outputs:
        all_files: list of all IBW file paths
        all_labels: list of corresponding labels
        all_scan_indices: list of scan indices
        dataset_info: dict with dataset metadata
    """

    all_files = []
    all_labels = []
    all_scan_indices = []
    dataset_info = {}
    
    for base_folder in base_folders:
        for dataset_name in dataset_names:
            dataset_path = os.path.join(base_folder, dataset_name)
            
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path {dataset_path} does not exist, skipping...")
                continue
                
            # Get all IBW files for this dataset
            ibw_files = sorted([
                os.path.join(dataset_path, f) 
                for f in os.listdir(dataset_path) 
                if f.lower().endswith(".ibw")
            ])
            
            if not ibw_files:
                print(f"Warning: No IBW files found in {dataset_path}, skipping...")
                continue
            
            print(f"Found {len(ibw_files)} IBW files in {dataset_path}")
            
            # Generate scan indices for this dataset (1-based)
            scan_indices = list(range(1, len(ibw_files) + 1))
            
            # Generate labels based on scan indices
            labels = generate_labels_from_scan_indices(scan_indices)
            
            # Store dataset info
            dataset_key = f"{os.path.basename(base_folder)}_{dataset_name}"
            dataset_info[dataset_key] = {
                'files': ibw_files,
                'labels': labels,
                'scan_indices': scan_indices,
                'count': len(ibw_files)
            }
            
            # Add to master lists
            all_files.extend(ibw_files)
            all_labels.extend(labels)
            all_scan_indices.extend(scan_indices)
    
    print(f"\nTotal files loaded: {len(all_files)}")
    print(f"Dataset breakdown:")
    for key, info in dataset_info.items():
        print(f"  {key}: {info['count']} files")
    
    return all_files, all_labels, all_scan_indices, dataset_info

class AugmentedTransform:
    """
    Custom transform class that handles the comprehensive data augmentation
    """
    # Crop size is defined as 3x3 image, but this can be changed via the crop_size arg
    def __init__(self, base_size=224, crop_size=86, normalize=True):
        self.base_size = base_size
        self.crop_size = crop_size
        self.normalize = normalize
        
        # Normalization, defined experimentally (can change)
        if normalize:
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize_transform = None
    
    def __call__(self, pil_image):
        """
        Apply the full augmentation pipeline to a single PIL image
        Returns a list of transformed tensors
        """
        # First resize to base size
        resize_transform = transforms.Resize((self.base_size, self.base_size))
        resized_img = resize_transform(pil_image)
        
        augmented_images = []
        
        # Augmentation profile: used to create 72 images from 1

        # Step 1: Create 9 crops (3x3 grid)
        crops = self._create_crops(resized_img)
        
        # Step 2: For each crop, create 4 rotations
        rotated_crops = []
        for crop in crops:
            for angle in [0, 90, 180, 270]:
                rotated = transforms.functional.rotate(crop, angle)
                rotated_crops.append(rotated)
        
        # Step 3: For each rotated crop, create original + horizontal flip
        for rotated_crop in rotated_crops:
            # Original
            tensor_orig = self._to_tensor_and_normalize(rotated_crop)
            augmented_images.append(tensor_orig)
            
            # Horizontally flipped
            flipped = transforms.functional.hflip(rotated_crop)
            tensor_flipped = self._to_tensor_and_normalize(flipped)
            augmented_images.append(tensor_flipped)
        
        return augmented_images
    
    def _create_crops(self, img):
        """Create 9 crops from the image in a 3x3 grid"""
        crops = []
        w, h = img.size
        
        # Calculate crop positions for 3x3 grid
        crop_w = self.crop_size
        crop_h = self.crop_size
        
        # Evenly distribute crops across the image
        x_positions = [0, (w - crop_w) // 2, w - crop_w]
        y_positions = [0, (h - crop_h) // 2, h - crop_h]
        
        for y in y_positions:
            for x in x_positions:
                crop = img.crop((x, y, x + crop_w, y + crop_h))
                # Resize crop back to base_size for consistency
                crop = crop.resize((self.base_size, self.base_size))
                crops.append(crop)
        
        return crops
    
    def _to_tensor_and_normalize(self, pil_img):
        """Convert PIL image to tensor and apply normalization"""
        tensor = transforms.ToTensor()(pil_img)
        if self.normalize_transform:
            tensor = self.normalize_transform(tensor)
        return tensor

class AugmentedIBWDataset(Dataset):
    def __init__(self, ibw_files, labels=None, scan_indices=None, 
                 compute_rewards=False, use_augmentation=True, 
                 augmentation_factor=72, 
                 reward_weights={
                        'height_consistency': 0.25,
                        'phase_consistency': 0.25, 
                        'sharpness': 0.15,
                        'snr': 0.15,
                        'data_diversity': 0.01,
                        'tip_freshness': 0.08,
                        'scan_rate': 0.02
                } ):
        """
        Inputs:
            ibw_files: list of IBW file paths
            labels: list of labels for each file
            scan_indices: list of scan indices for each file
            compute_rewards: whether to compute rewards
            use_augmentation: whether to apply data augmentation
            augmentation_factor: number of augmented images per original (default 72, as per above)
            reward_weights: list of weights for reward model, defaulted to experimental values
        """
        self.ibw_files = ibw_files
        self.labels = labels if labels is not None else [0] * len(ibw_files)
        self.scan_indices = scan_indices if scan_indices is not None else list(range(1, len(ibw_files) + 1))
        self.compute_rewards = compute_rewards
        self.use_augmentation = use_augmentation
        self.augmentation_factor = augmentation_factor

        # Valid indices (all indices by default)
        self.valid_indices = list(range(len(self.ibw_files)))

        # IBW cache (optional, not used unless implemented)
        self.ibw_cache = {}

        # Initialize augmentation transform
        if use_augmentation:
            self.augment_transform = AugmentedTransform(base_size=224, crop_size=75)
        
        # Pre-compute rewards if needed
        if compute_rewards:
            print("Computing rewards for all images...")
            self.rewards = []
            for i, file_path in enumerate(ibw_files):
                ibw_data = load_ibw(file_path)
                reward = compute_reward(ibw_data, scan_index=self.scan_indices[i], weights=reward_weights)
                self.rewards.append(reward)
            
            # Normalize rewards
            self.rewards = np.array(self.rewards)
            self.reward_mean = np.mean(self.rewards)
            self.reward_std = np.std(self.rewards)
            if self.reward_std > 0:
                self.rewards = (self.rewards - self.reward_mean) / self.reward_std
            
            print(f"Reward normalization: mean={self.reward_mean:.4f}, std={self.reward_std:.4f}")

            print(f"Given reward weights: {reward_weights}")

    def __len__(self):
        if self.use_augmentation:
            return len(self.ibw_files) * self.augmentation_factor
        else:
            return len(self.ibw_files)

    def __getitem__(self, idx):
        try:
            if self.use_augmentation:
                # Calculate which original image and which augmentation
                original_idx_pos = idx // self.augmentation_factor
                aug_idx = idx % self.augmentation_factor
            else:
                original_idx_pos = idx
                aug_idx = 0
            
            # Get the actual file index
            original_idx = self.valid_indices[original_idx_pos]
            
            # Load the IBW data
            if self.compute_rewards and original_idx in self.ibw_cache:
                ibw_data = self.ibw_cache[original_idx]
            else:
                file_path = self.ibw_files[original_idx]
                ibw_data = load_ibw(file_path)
                if ibw_data is None:
                    # Return a fallback item
                    return self._get_fallback_item(original_idx)
            
            height_img = ibw_data.z
            
            # Convert to PIL image
            pil_img = self._height_to_pil(height_img)
            
            if self.use_augmentation:
                # Get all augmented versions
                augmented_tensors = self.augment_transform(pil_img)
                if aug_idx < len(augmented_tensors):
                    img_tensor = augmented_tensors[aug_idx]
                else:
                    img_tensor = augmented_tensors[0]  # Fallback
            else:
                # Simple preprocessing without augmentation
                img_tensor = self._simple_preprocess(height_img)
            
            result = [img_tensor]
            
            # Add label (same for all augmentations of the same image)
            if self.labels is not None:
                result.append(self.labels[original_idx])
            
            # Add reward (same for all augmentations of the same image)
            if self.compute_rewards:
                result.append(self.rewards[original_idx])
            
            # Add scan index
            result.append(self.scan_indices[original_idx])
            
            return tuple(result)
            
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            traceback.print_exc()
            return self._get_fallback_item(0)

    def _get_fallback_item(self, idx):
        # Return a dummy tensor and default values for label, reward, scan_index
        dummy_tensor = torch.zeros(3, 224, 224)
        label = 0
        reward = 0.0
        scan_index = 0
        result = [dummy_tensor]
        if self.labels is not None:
            result.append(label)
        if self.compute_rewards:
            result.append(reward)
        result.append(scan_index)
        return tuple(result)

    def _height_to_pil(self, height_np_array):
        """Convert height numpy array to PIL Image"""
        # Normalize to 0-255
        h_min = np.min(height_np_array)
        h_max = np.max(height_np_array)
        norm_img = 255 * (height_np_array - h_min) / (h_max - h_min + 1e-8)
        norm_img = norm_img.astype(np.uint8)
        
        # Convert to PIL image and make RGB
        pil_img = Image.fromarray(norm_img)
        pil_img = pil_img.convert("RGB")
        
        return pil_img

def custom_collate(batch):
    """
    Handle variable batch contents based on what's included.
    """
    batch_size = len(batch)
    item_count = len(batch[0])
    
    imgs = torch.stack([item[0] for item in batch], dim=0)
    
    result = [imgs]
    
    if item_count >= 2:  # has labels
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        result.append(labels)
    
    if item_count >= 3:  # has rewards
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        result.append(rewards)
    
    if item_count >= 4:  # has scan indices
        scan_indices = torch.tensor([item[-1] for item in batch], dtype=torch.long)
        result.append(scan_indices)
    
    return tuple(result)

def preprocess_height_channel(height_np_array, transform=None):
    """
    Normalize the height channel to [0, 255] based on magnitude, convert to PIL Image,
    duplicate single channel to 3 channels, apply transforms.
    """
    h_min = np.min(height_np_array)
    h_max = np.max(height_np_array)
    norm_img = 255 * (height_np_array - h_min) / (h_max - h_min + 1e-8)
    norm_img = norm_img.astype(np.uint8)

    # Convert to PIL image
    pil_img = Image.fromarray(norm_img)

    # Convert grayscale to RGB by duplicating channels
    pil_img = pil_img.convert("RGB")

    if transform:
        return transform(pil_img)
    else:
        # Default tensor transform if none given
        return transforms.ToTensor()(pil_img)

def compute_reward(ibw_data,
                    scan_index=0, 
                    weights = {
                        'height_consistency': 0.25,
                        'phase_consistency': 0.25, 
                        'sharpness': 0.15,
                        'snr': 0.15,
                        'data_diversity': 0.01,
                        'tip_freshness': 0.08,
                        'scan_rate': 0.02
                        }, 
                    normalize_rewards=True):
    """
    Improved reward function with better scaling and additional quality metrics.

    Configure the weights away from experimentally-determined weights as needed. As is shown, the most
    important metrics are given above.
    """
    ch_names = ibw_data.channels
    
    # Extract data
    height_trace = ibw_data.data[ch_names.index('Height (T)')] 
    height_retrace = ibw_data.data[ch_names.index('Height (RT)')]
    phase_trace = ibw_data.data[ch_names.index('Phase (T)')] 
    phase_retrace = ibw_data.data[ch_names.index('Phase (RT)')]
    
    # Extract scan rate
    scan_rate = ibw_data.header.get('ScanRate', 1.0)
    
    rewards = {}
    
    # 1. Height Consistency (Trace vs Retrace)
    min_rows = min(height_trace.shape[0], height_retrace.shape[0])
    if min_rows > 0:
        height_diff = height_trace[:min_rows, :] - height_retrace[:min_rows, :]
        mae_height = np.mean(np.abs(height_diff))
        # Convert to similarity score (0 to 1, where 1 is perfect match)
        # Use exponential decay instead of inverse
        rewards['height_consistency'] = np.exp(-mae_height / np.std(height_trace))
    else:
        rewards['height_consistency'] = 0.0
    
    # 2. Phase Consistency (Trace vs Retrace)  
    min_rows_phase = min(phase_trace.shape[0], phase_retrace.shape[0])
    if min_rows_phase > 0:
        phase_diff = phase_trace[:min_rows_phase, :] - phase_retrace[:min_rows_phase, :]
        phase_std = np.std(phase_diff)
        # Normalize by the dynamic range of phase data
        phase_range = np.ptp(phase_trace)  # peak-to-peak
        if phase_range > 0:
            normalized_phase_std = phase_std / phase_range
            rewards['phase_consistency'] = np.exp(-5 * normalized_phase_std)  # More sensitive
        else:
            rewards['phase_consistency'] = 1.0
    else:
        rewards['phase_consistency'] = 0.0
    
    # 3. Image Sharpness/Focus (using gradient variance)
    height_gradients = np.gradient(height_trace)
    gradient_variance = np.var(height_gradients)
    # Normalize and convert to 0-1 score
    rewards['sharpness'] = np.tanh(gradient_variance / 1000.0)  # Adjust scaling as needed
    
    # 4. Signal-to-Noise Ratio
    height_signal = np.mean(height_trace)
    height_noise = np.std(height_trace)
    if height_noise > 0:
        snr = abs(height_signal) / height_noise
        rewards['snr'] = np.tanh(snr / 10.0)  # Normalize SNR
    else:
        rewards['snr'] = 1.0
    
    # 5. Data Quality (check for artifacts, saturation, etc.)
    # Detect if data is clipped/saturated
    height_flat = height_trace.flatten()
    unique_values = len(np.unique(height_flat))
    total_pixels = len(height_flat)
    diversity_ratio = unique_values / total_pixels
    rewards['data_diversity'] = min(diversity_ratio * 10, 1.0)  # Scale to 0-1
    
    # 6. Scan Index Penalty (prefer earlier scans, indicating tip quality)
    # Use exponential decay instead of linear
    max_expected_scans = 35  # Adjust based on your typical experiment
    rewards['tip_freshness'] = np.exp(-scan_index / max_expected_scans)
    
    # 7. Scan Rate Appropriateness (if you have an optimal range)
    optimal_scan_rate = 2.0  # Hz, adjust based on your system
    scan_rate_penalty = abs(scan_rate - optimal_scan_rate) / optimal_scan_rate
    rewards['scan_rate'] = np.exp(-scan_rate_penalty)
    
    # Weighted combination
    total_reward = sum(weights[key] * rewards[key] for key in weights.keys())
    
    return total_reward

class RewardAwareModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.2):
        super(RewardAwareModel, self).__init__()


        
        # Base model
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get feature dimension
        feature_dim = self.backbone.fc.in_features
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Reward prediction head
        self.reward_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, return_features=False):
        # Extract features
        features = self.features(x)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        
        # Classification output
        class_output = self.classifier(features)
        
        # Reward prediction output
        reward_output = self.reward_predictor(features)
        
        if return_features:
            return class_output, reward_output, features
        else:
            return class_output, reward_output

def validate_model(model, val_dataloader, device, use_adaptive_loss=False, criterion=None):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            if len(batch_data) != 4:
                continue
                
            inputs, labels, rewards, scan_indices = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            rewards = rewards.to(device)
            
            class_outputs, reward_outputs = model(inputs)
            
            if use_adaptive_loss and criterion:
                loss, _, _ = criterion(class_outputs, labels, reward_outputs, rewards)
            else:
                class_loss = F.cross_entropy(class_outputs, labels)
                reward_loss = F.mse_loss(reward_outputs.squeeze(), rewards)
                loss = class_loss + 0.1 * reward_loss
            
            total_loss += loss.item()
    
    return total_loss / len(val_dataloader)

def train_hybrid_model(model, train_dataloader, val_dataloader, optimizer, device, 
                        num_epochs=10, use_adaptive_loss=True):
    
    if use_adaptive_loss:
        criterion = AdaptiveLoss().to(device)
        # Include adaptive loss parameters in optimizer
        optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), 
                              lr=1e-4, weight_decay=1e-5)
    else:
        classification_criterion = nn.CrossEntropyLoss()
        reward_criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_reward_loss = 0.0
        
        for i, batch_data in enumerate(train_dataloader):
            if len(batch_data) != 4:  # Need all components
                continue
                
            inputs, labels, rewards, scan_indices = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            rewards = rewards.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_outputs, reward_outputs = model(inputs)
            
            if use_adaptive_loss:
                total_loss, class_loss, reward_loss = criterion(
                    class_outputs, labels, reward_outputs, rewards)
            else:
                class_loss = classification_criterion(class_outputs, labels)
                reward_loss = reward_criterion(reward_outputs.squeeze(), rewards)
                total_loss = class_loss + 0.1 * reward_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += total_loss.item()
            running_class_loss += class_loss.item()
            running_reward_loss += reward_loss.item()
            
            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}")
        
        # Validation phase
        val_loss = validate_model(model, val_dataloader, device, use_adaptive_loss, criterion if use_adaptive_loss else None)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Statistics
        avg_train_loss = running_loss / len(train_dataloader)
        avg_class_loss = running_class_loss / len(train_dataloader)
        avg_reward_loss = running_reward_loss / len(train_dataloader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Class Loss: {avg_class_loss:.4f} | Reward Loss: {avg_reward_loss:.4f}")
        
        if use_adaptive_loss:
            print(f"  Adaptive weights - Class: {torch.exp(-criterion.log_var_class).item():.4f}, "
                  f"Reward: {torch.exp(-criterion.log_var_reward).item():.4f}")
        
        # Save best model if it is better than previous
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_hybrid_model.pth')
        
        print("  Saved current epoch's model to 'best_hybrid_model.pth'.")

        print("-" * 50)
    
    return train_losses, val_losses

def evaluate_hybrid_model(model, dataloader, device):
    """Evaluate the model and return metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_rewards_true = []
    all_rewards_pred = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 4:  # imgs, labels, rewards, scan_indices
                inputs, labels, rewards, scan_indices = batch_data
            elif len(batch_data) == 3:  # imgs, labels, scan_indices
                inputs, labels, scan_indices = batch_data
                rewards = None
            else:
                continue  # Need at least labels for evaluation
                
            inputs = inputs.to(device)
            
            class_outputs, reward_outputs = model(inputs)
            
            # Get predictions
            _, predicted = torch.max(class_outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if rewards is not None:
                all_rewards_true.extend(rewards.numpy())
                all_rewards_pred.extend(reward_outputs.cpu().numpy().flatten())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Classification Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    
    if all_rewards_true:
        reward_mse = np.mean((np.array(all_rewards_true) - np.array(all_rewards_pred))**2)
        print(f"Reward Prediction MSE: {reward_mse:.4f}")
    
    return accuracy

class RealTimeHybridPlotter:
    """Enhanced real-time plotter for hybrid model training metrics"""
    
    def __init__(self, use_jupyter=True, window_size=100, update_frequency=1, scroll_size=1000):
        self.use_jupyter = use_jupyter
        self.window_size = window_size
        self.update_frequency = update_frequency  # Update plot every N batches
        self.update_counter = 0
        
        # Storage for metrics
        # self.batch_numbers = []
        # self.total_losses = []
        # self.class_losses = []
        # self.reward_losses = []
        self.batch_numbers = deque(maxlen=scroll_size)
        self.total_losses = deque(maxlen=scroll_size)
        self.class_losses = deque(maxlen=scroll_size)
        self.reward_losses = deque(maxlen=scroll_size)
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_numbers = []
        
        # Storage for running averages (smoother plotting)
        self.running_total_loss = deque(maxlen=10)
        self.running_class_loss = deque(maxlen=10)
        self.running_reward_loss = deque(maxlen=10)
        
        # Setup plotting
        if use_jupyter:
            plt.ion()  # Turn on interactive mode
            # Use a different backend that works better with Jupyter
            plt.switch_backend('module://ipykernel.pylab.backend_inline')
        else:
            plt.ion()
        
        # Create figure with better settings for real-time updates
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Hybrid Model Training Progress (Real-Time)', fontsize=16, fontweight='bold')
        
        # Create subplots with better spacing
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0, :])  # Full width for batch losses
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Validation loss
        self.ax3 = self.fig.add_subplot(gs[1, 1])  # Validation accuracy
        self.ax4 = self.fig.add_subplot(gs[2, :])  # Combined metrics
        
        # Initialize batch loss plot (main real-time plot)
        self.line_total, = self.ax1.plot([], [], 'b-', label='Total Loss', linewidth=2, alpha=0.8)
        self.line_class, = self.ax1.plot([], [], 'r-', label='Classification Loss', linewidth=2, alpha=0.8)
        self.line_reward, = self.ax1.plot([], [], 'g-', label='Reward Loss', linewidth=2, alpha=0.8)
        
        # Add smoothed lines
        self.line_total_smooth, = self.ax1.plot([], [], 'b--', label='Total (Smoothed)', linewidth=1, alpha=0.6)
        self.line_class_smooth, = self.ax1.plot([], [], 'r--', label='Class (Smoothed)', linewidth=1, alpha=0.6)
        self.line_reward_smooth, = self.ax1.plot([], [], 'g--', label='Reward (Smoothed)', linewidth=1, alpha=0.6)
        
        self.ax1.set_title('Training Losses - Real-Time (Per Batch)', fontweight='bold')
        self.ax1.set_xlabel('Batch Number')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend(loc='upper right', fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_yscale('log')  # Log scale often better for loss visualization
        
        # Validation loss plot
        self.line_val_loss, = self.ax2.plot([], [], 'purple', marker='o', linewidth=3, markersize=8)
        self.ax2.set_title('Validation Loss (Per Epoch)', fontweight='bold')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Validation Loss')
        self.ax2.grid(True, alpha=0.3)
        
        # Validation accuracy plot
        self.line_val_acc, = self.ax3.plot([], [], 'orange', marker='s', linewidth=3, markersize=8)
        self.ax3.set_title('Validation Accuracy (Per Epoch)', fontweight='bold')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Accuracy (%)')
        self.ax3.grid(True, alpha=0.3)
        
        # Combined epoch metrics with dual y-axis
        self.ax4_twin = self.ax4.twinx()
        self.line_epoch_loss, = self.ax4.plot([], [], 'purple', marker='o', linewidth=3, 
                                            markersize=8, label='Validation Loss')
        self.line_epoch_acc, = self.ax4_twin.plot([], [], 'orange', marker='s', linewidth=3, 
                                                markersize=8, label='Validation Accuracy')
        self.ax4.set_title('Validation Metrics - Combined View', fontweight='bold')
        self.ax4.set_xlabel('Epoch')
        self.ax4.set_ylabel('Validation Loss', color='purple', fontweight='bold')
        self.ax4_twin.set_ylabel('Validation Accuracy (%)', color='orange', fontweight='bold')
        self.ax4.tick_params(axis='y', labelcolor='purple')
        self.ax4_twin.tick_params(axis='y', labelcolor='orange')
        self.ax4.grid(True, alpha=0.3)
        
        # Add combined legend
        lines1, labels1 = self.ax4.get_legend_handles_labels()
        lines2, labels2 = self.ax4_twin.get_legend_handles_labels()
        self.ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Add text box for current statistics
        self.stats_text = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes, 
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Show initial plot
        if use_jupyter:
            plt.show()
        else:
            plt.show(block=False)
            plt.draw()
        
        self.batch_counter = 0
        self.last_update_time = time.time()
        
        # Force initial draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _calculate_smooth_values(self, values, window=10):
        """Calculate smoothed values using moving average"""
        if len(values) < window:
            return list(values)
        else:
            smoothed = []
            values_list = list(values)
            for i in range(len(values_list)):
                start_idx = max(0, i - window + 1)
                smoothed.append(np.mean(values_list[start_idx:i+1]))
            return smoothed
    
    def update_batch_metrics(self, total_loss, class_loss, reward_loss, force_update=False):
        """Update real-time batch metrics with improved performance"""
        self.batch_counter += 1
        self.update_counter += 1

        # Store metrics
        self.batch_numbers.append(self.batch_counter)
        self.total_losses.append(total_loss)
        self.class_losses.append(class_loss)
        self.reward_losses.append(reward_loss)
        
        # Store for running averages
        self.running_total_loss.append(total_loss)
        self.running_class_loss.append(class_loss)
        self.running_reward_loss.append(reward_loss)
        
        # Only update plot every N batches or if forced (to improve performance)
        if self.update_counter >= self.update_frequency or force_update:
            self.update_counter = 0
            self._update_batch_plot()
    
    def _update_batch_plot(self):
        """Internal method to update the batch loss plot"""
        if len(self.batch_numbers) == 0:
            return
            
        batch_nums = list(self.batch_numbers)
        total_losses = list(self.total_losses)
        class_losses = list(self.class_losses)
        reward_losses = list(self.reward_losses)
        
        # Update main lines
        self.line_total.set_data(batch_nums, total_losses)
        self.line_class.set_data(batch_nums, class_losses)
        self.line_reward.set_data(batch_nums, reward_losses)
        
        # Update smoothed lines
        if len(batch_nums) > 5:  # Only show smoothed lines after some data
            smooth_total = self._calculate_smooth_values(total_losses)
            smooth_class = self._calculate_smooth_values(class_losses)
            smooth_reward = self._calculate_smooth_values(reward_losses)
            
            self.line_total_smooth.set_data(batch_nums, smooth_total)
            self.line_class_smooth.set_data(batch_nums, smooth_class)
            self.line_reward_smooth.set_data(batch_nums, smooth_reward)
        
        # Adjust axes dynamically
        if len(batch_nums) > 1:
            self.ax1.set_xlim(max(1, min(batch_nums)), max(batch_nums))
            
            # Use log scale limits
            all_losses = total_losses + class_losses + reward_losses
            min_loss = max(1e-6, min(all_losses))  # Avoid log(0)
            max_loss = max(all_losses)
            
            self.ax1.set_ylim(min_loss * 0.5, max_loss * 2.0)
        
        # Update statistics text
        if len(self.running_total_loss) > 0:
            current_stats = (
                f'Batch: {self.batch_counter}\n'
                f'Current - Total: {total_losses[-1]:.4f}, Class: {class_losses[-1]:.4f}, Reward: {reward_losses[-1]:.4f}\n'
                f'Avg (last 10) - Total: {np.mean(self.running_total_loss):.4f}, '
                f'Class: {np.mean(self.running_class_loss):.4f}, Reward: {np.mean(self.running_reward_loss):.4f}'
            )
            self.stats_text.set_text(current_stats)
        
        # Force update
        self._force_plot_update()
    
    def _force_plot_update(self):
        """Force the plot to update immediately"""
        try:
            if self.use_jupyter:
                # Jupyter-specific update method
                clear_output(wait=True)
                display(self.fig)
            else:
                # Standard matplotlib update
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.001)  # Very small pause to allow GUI update
        except Exception as e:
            print(f"Warning: Plot update failed: {e}")
    
    def update_epoch_metrics(self, epoch, val_loss, val_accuracy):
        self.epoch_numbers.append(epoch)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        
        val_accuracy_pct = val_accuracy * 100
        val_accuracies_pct = [acc * 100 for acc in self.val_accuracies]

        # Update validation loss plot
        self.line_val_loss.set_data(self.epoch_numbers, self.val_losses)
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Update validation accuracy plot
        self.line_val_acc.set_data(self.epoch_numbers, val_accuracies_pct)
        self.ax3.relim()
        self.ax3.autoscale_view()

        # Update combined plot
        self.line_epoch_loss.set_data(self.epoch_numbers, self.val_losses)
        self.line_epoch_acc.set_data(self.epoch_numbers, val_accuracies_pct)
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.ax4_twin.relim()
        self.ax4_twin.autoscale_view()

        self._force_plot_update()
        print(f"  Epoch {epoch} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy_pct:.2f}%")

def validate_model_with_accuracy(model, val_dataloader, device, use_adaptive_loss=False, criterion=None):
    """Enhanced validation function that returns both loss and accuracy"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            if len(batch_data) != 4:
                continue
                
            inputs, labels, rewards, scan_indices = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            rewards = rewards.to(device)
            
            class_outputs, reward_outputs = model(inputs)
            
            # Calculate loss
            if use_adaptive_loss and criterion:
                loss, _, _ = criterion(class_outputs, labels, reward_outputs, rewards)
            else:
                class_loss = F.cross_entropy(class_outputs, labels)
                reward_loss = F.mse_loss(reward_outputs.squeeze(), rewards)
                loss = class_loss + 0.1 * reward_loss
            
            total_loss += loss.item()
            
            # Collect predictions for accuracy calculation
            _, predicted = torch.max(class_outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

def train_hybrid_model_with_plotting(model, train_dataloader, val_dataloader, optimizer, device, 
                                   num_epochs=10, use_adaptive_loss=True, use_jupyter=True,
                                   plot_update_frequency=5, scroll_size=400):
    """
    Enhanced training function with improved real-time plotting
    
    Args:
        plot_update_frequency: Update plot every N batches (lower = more frequent updates but slower)
    """
    
    # Initialize enhanced plotter
    plotter = RealTimeHybridPlotter(
        use_jupyter=use_jupyter, 
        window_size=400,    # Larger window for better visualization
        update_frequency=plot_update_frequency,
        scroll_size=scroll_size     # Define the scroll window size here
    )
    
    # Setup loss function and optimizer
    if use_adaptive_loss:
        # Assuming AdaptiveLoss is defined elsewhere
        try:
            criterion = AdaptiveLoss().to(device)
            optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), 
                                  lr=1e-4, weight_decay=1e-5)
        except NameError:
            print("AdaptiveLoss not found, using standard loss functions")
            use_adaptive_loss = False
            classification_criterion = nn.CrossEntropyLoss()
            reward_criterion = nn.MSELoss()
    else:
        classification_criterion = nn.CrossEntropyLoss()
        reward_criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"  Starting hybrid model training for {num_epochs} epochs...")
    print(f"  Plot updates every {plot_update_frequency} batches")
    print("=" * 80)
    
    try:
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_class_loss = 0.0
            running_reward_loss = 0.0
            batch_count = 0
            
            for i, batch_data in enumerate(train_dataloader):
                if len(batch_data) != 4:  # Need all components
                    continue
                    
                inputs, labels, rewards, scan_indices = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                rewards = rewards.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                class_outputs, reward_outputs = model(inputs)
                
                if use_adaptive_loss:
                    total_loss, class_loss, reward_loss = criterion(
                        class_outputs, labels, reward_outputs, rewards)
                else:
                    class_loss = classification_criterion(class_outputs, labels)
                    reward_loss = reward_criterion(reward_outputs.squeeze(), rewards)
                    total_loss = class_loss + 0.1 * reward_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update running statistics
                running_loss += total_loss.item()
                running_class_loss += class_loss.item()
                running_reward_loss += reward_loss.item()
                batch_count += 1
                
                # Update real-time plots
                plotter.update_batch_metrics(
                    total_loss.item(), 
                    class_loss.item(), 
                    reward_loss.item()
                )
                
                # Print progress periodically
                if (i + 1) % 100 == 0:
                    avg_loss = running_loss / batch_count
                    avg_class = running_class_loss / batch_count
                    avg_reward = running_reward_loss / batch_count
                    
                    print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_dataloader)}] - "
                          f"Loss: {total_loss.item():.4f} | Avg: {avg_loss:.4f} "
                          f"(Class: {avg_class:.4f}, Reward: {avg_reward:.4f})")
            
            # Force plot update at end of epoch
            plotter.update_batch_metrics(
                total_loss.item(), class_loss.item(), reward_loss.item(), force_update=True
            )
            
            # Validation phase
            print(f"  Running validation...")
            val_loss, val_accuracy = validate_model_with_accuracy(
                model, val_dataloader, device, use_adaptive_loss, 
                criterion if use_adaptive_loss else None
            )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Calculate epoch statistics
            avg_train_loss = running_loss / len(train_dataloader)
            avg_class_loss = running_class_loss / len(train_dataloader)
            avg_reward_loss = running_reward_loss / len(train_dataloader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Update epoch-level plots
            plotter.update_epoch_metrics(epoch + 1, val_loss, val_accuracy)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\n  Epoch {epoch+1}/{num_epochs} Summary ({epoch_time:.1f}s):")
            print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"  Class Loss: {avg_class_loss:.6f} | Reward Loss: {avg_reward_loss:.6f}")
            print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            
            if use_adaptive_loss:
                print(f"  Adaptive weights - Class: {torch.exp(-criterion.log_var_class).item():.4f}, "
                      f"Reward: {torch.exp(-criterion.log_var_reward).item():.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                }, 'best_hybrid_model.pth')
                
                print(f"NEW BEST MODEL SAVED! (Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f})")
                
                # Save plot of best model
                # plotter.save_plot(f'best_model_epoch_{epoch+1}.png')
            
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # print("Saving current progress...")
        # plotter.save_plot('interrupted_training.png')
    except Exception as e:
        print(f"Error during training: {e}")
        # plotter.save_plot('error_training.png')
        raise
    finally:
        if not use_jupyter:
            print("Training completed. Plot window will remain open...")
            print("Close the plot window manually when done reviewing.")
        
        # Save final plot
        # plotter.save_plot('final_training_progress.png')
    
    print(f"\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"Final plot saved as 'final_training_progress.png'")
    
    return train_losses, val_losses, val_accuracies, plotter

def load_trained_hybrid_model(model_path, num_classes=5, device='cpu'):
    """
    Load a trained model from saved checkpoint
    
    Inputs:
        model_path: Path to saved model (.pth file)
        num_classes: Number of quality classes
        device: PyTorch device
    
    Outputs:
        Loaded model ready for inference
    """
    model = RewardAwareModel(num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def hb_predict_ibw(model, ibw_file_path, transform, device, 
                   reward_mean=None, reward_std=None, scan_index=None):
    """
    Predict quality class and reward for a single IBW file using all augmentations.
    """
    import warnings
    model.eval()
    try:
        # Load and preprocess the IBW file
        ibw_data = load_ibw(ibw_file_path)
        height_img = ibw_data.z

        # Convert to PIL image (if not already)
        from PIL import Image
        h_min = np.min(height_img)
        h_max = np.max(height_img)
        norm_img = 255 * (height_img - h_min) / (h_max - h_min + 1e-8)
        norm_img = norm_img.astype(np.uint8)
        pil_img = Image.fromarray(norm_img).convert("RGB")

        # Apply augmentation transform (returns a list of tensors)
        augmented_tensors = transform(pil_img)
        if isinstance(augmented_tensors, torch.Tensor):
            augmented_tensors = [augmented_tensors]

        if not augmented_tensors:
            warnings.warn(f"No augmentations produced for {ibw_file_path}.")
            return {
                'file_path': ibw_file_path,
                'error': 'No augmentations produced.',
                'predicted_class': None
            }

        # Stack into a batch
        batch_tensor = torch.stack(augmented_tensors).to(device)

        with torch.no_grad():
            class_outputs, reward_outputs = model(batch_tensor)
            probabilities = torch.softmax(class_outputs, dim=1)
            avg_probabilities = probabilities.mean(dim=0)
            avg_reward_normalized = reward_outputs.mean().item()

            # Check for NaNs in outputs
            if (torch.isnan(avg_probabilities).any() or 
                np.isnan(avg_reward_normalized)):
                warnings.warn(f"NaN encountered in prediction for {ibw_file_path}.")
                return {
                    'file_path': ibw_file_path,
                    'error': 'NaN encountered in prediction.',
                    'predicted_class': None
                }

            predicted_class = torch.argmax(avg_probabilities).item()
            confidence = avg_probabilities[predicted_class].item()

            if reward_mean is not None and reward_std is not None and reward_std > 0:
                predicted_reward_original = avg_reward_normalized * reward_std + reward_mean
            else:
                predicted_reward_original = avg_reward_normalized

            if scan_index is not None:
                actual_reward = compute_reward(ibw_data, scan_index=scan_index)
            else:
                actual_reward = compute_reward(ibw_data, scan_index=0)

        quality_descriptions = {
            0: "Excellent (Class 0)",
            1: "Good (Class 1)", 
            2: "Fair (Class 2)",
            3: "Poor (Class 3)",
            4: "Bad (Class 4)"
        }

        result = {
            'file_path': ibw_file_path,
            'predicted_class': predicted_class,
            'quality_description': quality_descriptions[predicted_class],
            'confidence': confidence,
            'class_probabilities': avg_probabilities.cpu().numpy(),
            'predicted_reward_normalized': avg_reward_normalized,
            'predicted_reward_original': predicted_reward_original,
            'actual_reward': actual_reward,
            'reward_difference': abs(predicted_reward_original - actual_reward),
            'scan_index_used': scan_index if scan_index is not None else 0,
            'num_augmentations': len(augmented_tensors)
        }
        return result

    except Exception as e:
        import warnings
        print(f"Prediction failed for {ibw_file_path}: {e}")
        return {
            'file_path': ibw_file_path,
            'error': str(e),
            'predicted_class': None
        }

def hb_predict_with_metadata(model, ibw_file_paths, model_metadata_path, device):
    """
    Predict using the same normalization parameters as training
    
    Inputs:
        model: Trained model
        ibw_file_paths: List of IBW file paths
        model_metadata_path: Path to saved training metadata (pickle file)
        device: PyTorch device
    
    Outputs:
        List of prediction results
    """
    import pickle
    
    # Load training metadata
    try:
        with open(model_metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        transform = metadata['transform']
        reward_mean = metadata.get('reward_mean', None)
        reward_std = metadata.get('reward_std', None)
        file_to_index = metadata.get('file_to_index', {})  # Maps file paths to scan indices
        
        print(f"Loaded training metadata:")
        print(f"  Reward normalization - Mean: {reward_mean}, Std: {reward_std}")
        print(f"  Transform: {transform}")
        
    except FileNotFoundError:
        print("Warning: No metadata file found. Using default parameters.")
        # Use same transform as training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        reward_mean = reward_std = None
        file_to_index = {}
    
    results = []
    
    print(f"Predicting quality for {len(ibw_file_paths)} files...")
    
    for i, file_path in enumerate(ibw_file_paths):
        print(f"Processing file {i+1}/{len(ibw_file_paths)}: {file_path}")
        
        # Get scan index if available from training
        scan_index = extract_scan_index_from_filename(file_path)
        print(f"File: {os.path.basename(file_path)}, Extracted scan_index: {scan_index}")
        aug_transform = AugmentedTransform(base_size=256, crop_size=86, normalize=True)
        result = hb_predict_ibw(model, file_path, aug_transform, device, scan_index=scan_index)
        results.append(result)
        
        if 'error' not in result:
            print(f"Scan Index: {scan_index} Predicted: Class {result['predicted_class']} ({result['quality_description']}) "
                  f"with {result['confidence']:.3f} confidence")
        else:
            print(f"  Error: {result['error']}")
    
    return results

def extract_scan_index_from_filename(file_path):
    """
    Extract scan index from Igor-style filenames
    
    Inputs:
        file_path: Path like "exp_data/June 18/wear_out/Wear_out_0015.ibw"
    
    Outputs:
        int: Scan index (e.g., 15 from "0015")
    """
    import re
    import os
    
    filename = os.path.basename(file_path)
    
    # Look for pattern like "_0015" or similar 4-digit numbers
    match = re.search(r'_(\d{4})', filename)
    if match:
        return int(match.group(1))
    
    # Fallback: look for any number sequence
    match = re.search(r'(\d+)', filename)
    if match:
        print(f"Found index from filename: {int(match.group())}")
        return int(match.group(1))
    
    # Default fallback
    print(f"Could not find index from filename.")
    return 0

def add_true_labels_to_results_using_scan_indices(results):
    """
    Extract scan indices from each result's filename and generate
    true labels using your generate_labels_from_scan_indices function.
    
    Inputs:
        results (list of dict): Each dict has 'file_path'
        
    Outputs:
        results (list of dict): Each dict updated with 'true_class'
    """
    # Extract all scan indices first
    scan_indices = []
    for r in results:
        file_path = r.get('file_path', None)
        if file_path is None:
            scan_indices.append(None)
        else:
            scan_indices.append(extract_scan_index_from_filename(file_path))
    
    # Filter out None indices before label generation
    valid_scan_indices = [idx for idx in scan_indices if idx is not None]
    
    # Generate labels only for valid indices
    generated_labels = generate_labels_from_scan_indices(valid_scan_indices)
    
    # Map generated labels back to results (handle None gracefully)
    label_idx = 0
    for i, idx in enumerate(scan_indices):
        if idx is None:
            results[i]['true_class'] = None
        else:
            results[i]['true_class'] = generated_labels[label_idx]
            label_idx += 1
    
    return results
    
def hb_generate_roc_auc_curve(results, num_classes=5, title="ROC Curve", output_path=None):
    """
    Generate and plot the ROC-AUC curve for multiclass classification.
    """
    y_true = []
    y_probs = []

    for r in results:
        if r.get("true_class") is not None:
            y_true.append(r["true_class"])
            y_probs.append(r["class_probabilities"])

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # Binarize the output for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)

    # if output_path:
    #     # plt.savefig(output_path)
    #     print(f"ROC curve saved to {output_path}")
    # else:
    #   plt.show()

    plt.show()

    return roc_auc

def hb_inspect_results(results, num_samples=5):
    """
    Inspect a few samples from the results list to understand their structure.

    Prints:
    - Keys available in each result dict
    - True label (if any)
    - Predicted class
    - Class probabilities and their sum

    Inputs:
        results (list of dict): Your prediction results
        num_samples (int): How many samples to show
    """
    print(f"Total results: {len(results)}")
    print("Inspecting first", num_samples, "samples...\n")

    for i, r in enumerate(results[:num_samples]):
        print(f"Sample #{i + 1}:")
        print("Keys:", list(r.keys()))

        # Try to print true label if exists
        true_label = r.get('true_class') or r.get('label') or r.get('target') or None
        print("True label:", true_label)

        pred_class = r.get('predicted_class')
        print("Predicted class:", pred_class)

        class_probs = r.get('class_probabilities')
        if class_probs is not None:
            print("Class probabilities:", class_probs)
            print("Sum of probabilities:", sum(class_probs))
        else:
            print("Class probabilities: None")

        print("-" * 40)

def generate_labels_from_scan_indices(scan_indices):
    sorted_indices = sorted(scan_indices)
    total_samples = len(sorted_indices)
    samples_per_group = total_samples / 5.0
    
    index_to_label = {}
    for i, idx in enumerate(sorted_indices):
        label = min(int(i / samples_per_group), 4)
        index_to_label[idx] = label
    
    labels = [index_to_label[idx] for idx in scan_indices]
    return labels

def get_all_ibw_files(folder, recursive=True):
    """Return a list of all .ibw files in a folder (optionally recursive)"""
    ibw_files = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith('.ibw'):
                    ibw_files.append(os.path.join(root, f))
    else:
        ibw_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.ibw')]
    return ibw_files

def hb_compute_confusion_matrix_from_multiple_folders(parent_folder, subfolders, model, num_classes=5, title="Confusion Matrix"):
    """
    Reads .ibw files from multiple subfolders, predicts classes, and plots the confusion matrix.
    True classes are assigned by splitting scan indices into equal groups.
    """
    all_files = []
    for sub in subfolders:
        folder_path = os.path.join(parent_folder, sub)
        files = get_all_ibw_files(folder_path)
        print(f"Found {len(files)} files in {folder_path}")
        all_files.extend(files)

    if not all_files:
        print(f"No .ibw files found in specified folders: {subfolders}")
        return None

    scan_indices = [extract_scan_index_from_filename(f) for f in all_files]
    
    # Sort files and scan indices together
    sorted_pairs = sorted(zip(scan_indices, all_files), key=lambda x: x[0])
    sorted_scan_indices, sorted_files = zip(*sorted_pairs)
    total_files = len(sorted_files)

    # True classes
    true_classes = generate_labels_from_scan_indices(sorted_scan_indices)

    # Predict classes
    predicted_classes = []

    # Dummy metadatapath, not supported for this prediction
    model_metadata_path = ""

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch, 'has_mps') and torch.has_mps and torch.mps.is_available() else 'cpu')

    results = hb_predict_with_metadata(
        model, list(sorted_files), model_metadata_path, device=device
    )
    predicted_classes = [r['predicted_class'] for r in results]
    
    # Ensure integer dtype
    true_classes = np.array(true_classes, dtype=int)
    predicted_classes = np.array(predicted_classes, dtype=int)

    print("Unique true labels:", np.unique(true_classes))
    print("Unique predicted labels:", np.unique(predicted_classes))

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(num_classes)])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()
    return cm

"""
Barlow Twins model training functions and class definitions for encoder training and a reward-aware classifier model.
"""

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_offdiag=0.0051):
        super().__init__()
        self.lambda_offdiag = lambda_offdiag

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        N, D = z1.size()
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)
        c = (z1.T @ z2) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_offdiag * off_diag

def height_to_pil(height_array):
    h_min, h_max = np.min(height_array), np.max(height_array)
    norm_img = 255 * (height_array - h_min) / (h_max - h_min + 1e-8)
    norm_img = norm_img.astype(np.uint8)
    pil_img = Image.fromarray(norm_img)
    return pil_img.convert("RGB")

class BT_AFMBarlowDataset(Dataset):
    def __init__(self, ibw_data_list, transform=None):
        self.data = ibw_data_list  # List of numpy arrays (256x256)
        self.transform = transform

    # Augmentation for this class is different than Hybrid model
    def __len__(self):
        return len(self.data) * 9  # 9 crops per image

    def __getitem__(self, idx):
        img_idx = idx // 9
        crop_idx = idx % 9

        height_img = self.data[img_idx]
        pil_img = height_to_pil(height_img)

        crops = self._create_crops(pil_img)
        base_crop = crops[crop_idx]

        # Generate two random augmentations of the same crop
        img1 = self.transform(base_crop)
        img2 = self.transform(base_crop)
        return img1, img2

    def _create_crops(self, img, crop_size=75):
        w, h = img.size
        positions = [0, (w - crop_size) // 2, w - crop_size]
        crops = []
        for y in positions:
            for x in positions:
                crop = img.crop((x, y, x + crop_size, y + crop_size))
                crops.append(crop.resize((224, 224)))
        return crops

def train_barlow_twins(model, dataloader, loss_fn, optimizer, device, epochs):
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for (x1, x2) in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)

            loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x1.size(0)

        avg_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}: Barlow Twins Loss = {avg_loss:.4f}")

        # Update best model if loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "best_barlow_twins_encoder.pth")
            print(f"Saved new best model at epoch {epoch} with loss {avg_loss:.4f}")

    return best_model_state

class EnhancedBT_AFMDataset(Dataset):
    def __init__(self, ibw_data_list, scan_indices, compute_rewards=False, transform=None, 
                 use_augmentation=True, crops_per_image=9, reward_weights={
                        'height_consistency': 0.25,
                        'phase_consistency': 0.25, 
                        'sharpness': 0.15,
                        'snr': 0.15,
                        'data_diversity': 0.01,
                        'tip_freshness': 0.08,
                        'scan_rate': 0.02
                } ):
        """
        Enhanced dataset that uses scan indices to generate multi-class labels
        
        Args:
            ibw_data_list: list of loaded IBW data objects
            scan_indices: list of scan indices corresponding to each IBW file
            compute_rewards: whether to compute quality rewards
            transform: torchvision transforms to apply
            use_augmentation: whether to use data augmentation (crops)
            crops_per_image: number of crops per image (default 9 for 3x3 grid)
        """
        self.data = ibw_data_list
        self.scan_indices = scan_indices
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.crops_per_image = crops_per_image
        
        # Generate multi-class labels from scan indices
        self.labels = generate_labels_from_scan_indices(scan_indices)
        
        print(f"Generated {len(self.labels)} labels from scan indices")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        # Compute rewards if requested
        self.rewards = None
        if compute_rewards:
            print("Computing rewards for all images...")
            self.rewards = []
            for i, data in enumerate(ibw_data_list):
                reward = compute_reward(data, scan_index=scan_indices[i], weights=reward_weights)
                self.rewards.append(reward)
            
            # Normalize rewards
            self.rewards = np.array(self.rewards)
            if len(self.rewards) > 1:
                self.reward_mean = np.mean(self.rewards)
                self.reward_std = np.std(self.rewards)
                if self.reward_std > 0:
                    self.rewards = (self.rewards - self.reward_mean) / self.reward_std
                print(f"Reward normalization: mean={self.reward_mean:.4f}, std={self.reward_std:.4f}")

    def __len__(self):
        if self.use_augmentation:
            return len(self.data) * self.crops_per_image
        else:
            return len(self.data)

    def __getitem__(self, idx):
        try:
            if self.use_augmentation:
                img_idx = idx // self.crops_per_image
                crop_idx = idx % self.crops_per_image
            else:
                img_idx = idx
                crop_idx = 0
            
            # Get height data
            height_img = self.data[img_idx].z
            label = self.labels[img_idx]
            
            # Convert to PIL image
            pil_img = height_to_pil(height_img)
            
            # Create crop if using augmentation
            if self.use_augmentation:
                crop = self._create_crop(pil_img, crop_idx)
            else:
                crop = pil_img.resize((224, 224))
            
            # Apply transforms
            if self.transform:
                crop = self.transform(crop)
            
            result = [crop, label]
            
            # Add reward if computed
            if self.rewards is not None:
                result.append(self.rewards[img_idx])
            
            # Add scan index
            result.append(self.scan_indices[img_idx])
            
            return tuple(result)
            
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            traceback.print_exc()
            # Return fallback
            dummy_tensor = torch.zeros(3, 224, 224)
            return (dummy_tensor, 0, 0.0, 1)

    def _create_crop(self, img, idx, crop_size=75):
        """Create crop from image using 3x3 grid pattern"""
        w, h = img.size
        positions = [0, (w - crop_size) // 2, w - crop_size]
        y, x = divmod(idx, 3)
        crop = img.crop((positions[x], positions[y], 
                        positions[x] + crop_size, positions[y] + crop_size))
        return crop.resize((224, 224))

class BarlowTwinsClassifier(nn.Module):
    def __init__(self, num_classes=5, include_reward_head=True, dropout_rate=0.65):
        super(BarlowTwinsClassifier, self).__init__()
        import torchvision.models as models
        # Encoder (ResNet18)
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        
        # Get feature dimension
        feature_dim = 512  # ResNet18 feature dimension
        
        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Optional reward prediction head
        self.include_reward_head = include_reward_head
        if include_reward_head:
            self.reward_predictor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        
        class_output = self.classifier(features)
        
        reward_output = None
        if self.include_reward_head:
            reward_output = self.reward_predictor(features)
        
        if return_features:
            return class_output, reward_output, features
        else:
            if self.include_reward_head:
                return class_output, reward_output
            else:
                return class_output

def bt_load_multiclass_classifier(model_path, device='cpu', num_classes=5):
    """Load the trained multiclass classifier"""
    checkpoint = torch.load(model_path, map_location=device)
    has_reward_head = any(k.startswith("reward_predictor") for k in checkpoint['model_state_dict'].keys())
    
    model = BarlowTwinsClassifier(
        num_classes=num_classes, 
        include_reward_head=has_reward_head,
        dropout_rate=0.3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return model

def bt_train_classifier(model, train_loader, val_loader, optimizer, device, 
                    num_epochs=50, use_reward_loss=False, reward_weight=0.1):
    """
    Enhanced training function with optional reward prediction
    """
    classification_criterion = nn.CrossEntropyLoss()
    reward_criterion = nn.MSELoss() if use_reward_loss else None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_reward_loss = 0.0
        correct = 0
        total = 0
        
        for i, batch_data in enumerate(train_loader):
            # Handle variable batch contents
            if len(batch_data) == 4:  # inputs, labels, rewards, scan_indices
                inputs, labels, rewards, scan_indices = batch_data
                has_rewards = True
            elif len(batch_data) == 3:  # inputs, labels, scan_indices
                inputs, labels, scan_indices = batch_data
                has_rewards = False
            else:
                continue
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            if has_rewards:
                if rewards.dtype != torch.float32:
                    rewards = rewards.float()

                rewards = rewards.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            if model.include_reward_head:
                class_outputs, reward_outputs = model(inputs)
            else:
                class_outputs = model(inputs)
                reward_outputs = None
            
            # Compute losses
            class_loss = classification_criterion(class_outputs, labels)
            total_loss = class_loss
            
            if use_reward_loss and reward_outputs is not None and has_rewards:
                reward_loss = reward_criterion(reward_outputs.squeeze(), rewards)
                total_loss += reward_weight * reward_loss
                running_reward_loss += reward_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            running_class_loss += class_loss.item()
            
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
        
        # Validation phase
        val_loss, val_accuracy = bt_evaluate_classifier(model, val_loader, device, 
                                                   use_reward_loss, reward_weight)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Statistics
        train_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_class_loss = running_class_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
        print(f"  Class Loss: {avg_class_loss:.4f}")
        
        if use_reward_loss and running_reward_loss > 0:
            avg_reward_loss = running_reward_loss / len(train_loader)
            print(f"  Reward Loss: {avg_reward_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, 'best_barlow_classifier.pth')
            print(f"  Saved best model (Val Loss: {val_loss:.4f})")
        
        print("-" * 60)
    
    return train_losses, val_losses

def bt_evaluate_classifier(model, dataloader, device, use_reward_loss=False, reward_weight=0.1):
    """Evaluate the classifier and return loss and accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    classification_criterion = nn.CrossEntropyLoss()
    reward_criterion = nn.MSELoss() if use_reward_loss else None
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle variable batch contents
            if len(batch_data) == 4:
                inputs, labels, rewards, scan_indices = batch_data
                has_rewards = True
            elif len(batch_data) == 3:
                inputs, labels, scan_indices = batch_data
                has_rewards = False
            else:
                continue
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            if has_rewards:
                if rewards.dtype != torch.float32:
                    rewards = rewards.float()

                rewards = rewards.to(device)
            
            # Forward pass
            if model.include_reward_head:
                class_outputs, reward_outputs = model(inputs)
            else:
                class_outputs = model(inputs)
                reward_outputs = None
            
            # Compute losses
            class_loss = classification_criterion(class_outputs, labels)
            loss = class_loss
            
            if use_reward_loss and reward_outputs is not None and has_rewards:
                reward_loss = reward_criterion(reward_outputs.squeeze(), rewards)
                loss += reward_weight * reward_loss
            
            total_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

class RealTimeLossPlotter:
    def __init__(self, max_points=1000, use_jupyter=True):
        self.max_points = max_points
        self.epochs = []
        self.losses = []
        self.use_jupyter = use_jupyter
        
        # Set up matplotlib backend
        if self.use_jupyter:
            # For Jupyter notebooks
            plt.style.use('default')
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
        else:
            # For regular Python scripts
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        self.setup_plot()
        
    def setup_plot(self):
        self.ax.set_xlabel('Epoch', fontsize=12)
        self.ax.set_ylabel('Loss', fontsize=12)
        self.ax.set_title('Barlow Twins Training Loss - Real Time', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
    def update_plot(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # Keep only the last max_points for memory efficiency
        if len(self.epochs) > self.max_points:
            self.epochs = self.epochs[-self.max_points:]
            self.losses = self.losses[-self.max_points:]
        
        # Clear and redraw (more reliable for Jupyter)
        self.ax.clear()
        self.setup_plot()
        
        # Plot the loss curve
        self.ax.plot(self.epochs, self.losses, 'b-', linewidth=2, label='Training Loss')
        
        # Add some styling
        if len(self.losses) > 1:
            # Add trend line for last 20 points if we have enough data
            if len(self.losses) >= 20:
                recent_epochs = self.epochs[-20:]
                recent_losses = self.losses[-20:]
                z = np.polyfit(recent_epochs, recent_losses, 1)
                p = np.poly1d(z)
                self.ax.plot(recent_epochs, p(recent_epochs), "r--", alpha=0.7, label='Recent Trend')
        
        # # Set axis limits with some padding
        # if len(self.epochs) > 1:
        #     x_min, x_max = min(self.epochs), max(self.epochs)
        #     y_min, y_max = min(self.losses), max(self.losses)
            
        #     x_padding = max(1, (x_max - x_min) * 0.05)
        #     y_padding = max(0.001, (y_max - y_min) * 0.1)
            
        #     self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
        #     self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Add current loss value as text
        if self.losses:
            current_loss = self.losses[-1]
            self.ax.text(0.02, 0.98, f'Current Loss: {current_loss:.6f}', 
                        transform=self.ax.transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.legend()
        
        if self.use_jupyter:
            # For Jupyter notebooks - clear output and redisplay
            clear_output(wait=True)
            display(self.fig)
        else:
            # For regular Python scripts
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to ensure update
    
    def close(self):
        if not self.use_jupyter:
            plt.ioff()
        plt.close(self.fig)

def train_barlow_twins_with_plotting(model, dataloader, loss_fn, optimizer, device, epochs=100, use_jupyter=True):
    """
    Modified training function that includes real-time loss plotting
    Args:
        use_jupyter: Set to True if running in Jupyter notebook, False otherwise
    """
    model.train()
    plotter = RealTimeLossPlotter(use_jupyter=use_jupyter)
    
    print(f"Starting training for {epochs} epochs...")
    print("=" * 50)
    
    best_loss = 0

    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (view1, view2) in enumerate(dataloader):
                view1, view2 = view1.to(device), view2.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                z1 = model(view1)
                z2 = model(view2)
                
                # Compute loss
                loss = loss_fn(z1, z2)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Print batch progress occasionally
                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.6f}')
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / num_batches

            # Store first epoch score to compare
            if (epoch == 1):
                best_loss = epoch_loss
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, "best_barlow_twins_encoder.pth")
                print(f"Saved new best model at epoch {epoch} with loss {avg_loss:.4f}")

            plotter.update_plot(epoch + 1, avg_loss)
            
            # Print epoch summary
            print(f'Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.6f}')
            print("-" * 30)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        if not use_jupyter:
            print("Training completed. Close the plot window to continue...")
            input("Press Enter to close the plot and continue...")
        plotter.close()

class RealTimeAccuracyPlotter:
    """
    Real-time plotter for training and validation accuracy during classifier training
    """
    def __init__(self, use_jupyter=True, max_points=None):
        self.use_jupyter = use_jupyter
        self.max_points = max_points  # None means no limit
        
        # Data storage - no maxlen if max_points is None
        if max_points is None:
            self.epochs = []
            self.train_accuracies = []
            self.val_accuracies = []
            self.train_losses = []
            self.val_losses = []
        else:
            self.epochs = deque(maxlen=max_points)
            self.train_accuracies = deque(maxlen=max_points)
            self.val_accuracies = deque(maxlen=max_points)
            self.train_losses = deque(maxlen=max_points)
            self.val_losses = deque(maxlen=max_points)
        
        # Setup plotting with resizable figure
        if use_jupyter:
            plt.ion()  # Interactive mode for Jupyter
        
        # Create resizable figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 12))
        self.fig.suptitle('Classifier Training Progress', fontsize=16, fontweight='bold')
        
        # Make figure resizable
        if hasattr(self.fig.canvas, 'toolbar_visible'):
            self.fig.canvas.toolbar_visible = True
        if hasattr(self.fig.canvas, 'resizable'):
            self.fig.canvas.resizable = True
        
        # Initialize accuracy plot
        self.ax1.set_title('Training & Validation Accuracy', fontweight='bold')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy (%)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(0, 100)
        
        # Initialize loss plot with auto-scaling
        self.ax2.set_title('Training & Validation Loss', fontweight='bold')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_autoscaley_on(True)  # Enable auto-scaling for Y axis
        
        # Line objects for accuracy
        self.train_acc_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
        self.val_acc_line, = self.ax1.plot([], [], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
        self.ax1.legend()
        
        # Line objects for loss
        self.train_loss_line, = self.ax2.plot([], [], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
        self.val_loss_line, = self.ax2.plot([], [], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
        self.ax2.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # Add space between subplots
        
        if not use_jupyter:
            # Create a resizable window
            mngr = plt.get_current_fig_manager()
            if hasattr(mngr, 'window'):
                if hasattr(mngr.window, 'wm_state'):
                    mngr.window.wm_state('normal')  # Make window resizable
            plt.show(block=False)
            
        # Enable interactive navigation
        plt.rcParams['figure.max_open_warning'] = 0
    
    def update_plot(self, epoch, train_acc, val_acc, train_loss, val_loss):
        """Update the plot with new accuracy and loss values"""
        self.epochs.append(epoch)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Convert to lists if using regular lists instead of deque
        epochs_list = list(self.epochs)
        train_acc_list = list(self.train_accuracies)
        val_acc_list = list(self.val_accuracies)
        train_loss_list = list(self.train_losses)
        val_loss_list = list(self.val_losses)
        
        # Update accuracy plot
        self.train_acc_line.set_data(epochs_list, train_acc_list)
        self.val_acc_line.set_data(epochs_list, val_acc_list)
        
        # Update loss plot
        self.train_loss_line.set_data(epochs_list, train_loss_list)
        self.val_loss_line.set_data(epochs_list, val_loss_list)
        
        # Adjust axes limits to show all data
        if len(epochs_list) > 0:
            # Accuracy plot limits - show all epochs
            self.ax1.set_xlim(min(epochs_list) - 1, max(epochs_list) + 1)
            min_acc = min(min(train_acc_list), min(val_acc_list))
            max_acc = max(max(train_acc_list), max(val_acc_list))
            acc_margin = (max_acc - min_acc) * 0.1
            self.ax1.set_ylim(max(0, min_acc - acc_margin), min(100, max_acc + acc_margin))
            
            # Loss plot limits - show all epochs
            self.ax2.set_xlim(min(epochs_list) - 1, max(epochs_list) + 1)
            min_loss = min(min(train_loss_list), min(val_loss_list))
            max_loss = max(max(train_loss_list), max(val_loss_list))
            loss_range = max_loss - min_loss
            loss_margin = loss_range * 0.1 if loss_range > 0 else 0.1
            self.ax2.set_ylim(min_loss - loss_margin, max_loss + loss_margin)
        
        # Add current values as text
        if len(epochs_list) > 0:
            latest_train_acc = train_acc_list[-1]
            latest_val_acc = val_acc_list[-1]
            latest_train_loss = train_loss_list[-1]
            latest_val_loss = val_loss_list[-1]
            
            # Clear previous text and add new
            for txt in self.ax1.texts[:]:
                txt.remove()
            for txt in self.ax2.texts[:]:
                txt.remove()
                
            # Add epoch count and current metrics
            self.ax1.text(0.02, 0.98, f'Epoch: {epoch} | Train Acc: {latest_train_acc:.2f}%', 
                         transform=self.ax1.transAxes, fontsize=11, fontweight='bold',
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
            self.ax1.text(0.02, 0.88, f'Val Acc: {latest_val_acc:.2f}%', 
                         transform=self.ax1.transAxes, fontsize=11, fontweight='bold',
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
            
            self.ax2.text(0.02, 0.98, f'Train Loss: {latest_train_loss:.4f}', 
                         transform=self.ax2.transAxes, fontsize=11, fontweight='bold',
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
            self.ax2.text(0.02, 0.88, f'Val Loss: {latest_val_loss:.4f}', 
                         transform=self.ax2.transAxes, fontsize=11, fontweight='bold',
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
        
        # Refresh the display
        if self.use_jupyter:
            from IPython.display import clear_output, display
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
    
    def close(self):
        """Close the plot"""
        plt.close(self.fig)
    
    # def save_plot(self, filename='classifier_training_progress.png'):
    #     """Save the current plot to file"""
    #     self.fig.savefig(filename, dpi=300, bbox_inches='tight')
    #     print(f"Plot saved as {filename}")

def bt_train_classifier_with_plotting(model, train_loader, val_loader, optimizer, device, 
                                    num_epochs=50, use_reward_loss=False, reward_weight=0.1, 
                                    use_jupyter=True):
    """
    Enhanced training function with real-time accuracy plotting
    """
    classification_criterion = nn.CrossEntropyLoss()
    reward_criterion = nn.MSELoss() if use_reward_loss else None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=5)
    
    # Initialize plotter
    plotter = RealTimeAccuracyPlotter(use_jupyter=use_jupyter)
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    
    print(f"Starting classifier training for {num_epochs} epochs...")
    print("=" * 60)
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            running_class_loss = 0.0
            running_reward_loss = 0.0
            correct = 0
            total = 0
            
            for i, batch_data in enumerate(train_loader):
                # Handle variable batch contents
                if len(batch_data) == 4:  # inputs, labels, rewards, scan_indices
                    inputs, labels, rewards, scan_indices = batch_data
                    has_rewards = True
                elif len(batch_data) == 3:  # inputs, labels, scan_indices
                    inputs, labels, scan_indices = batch_data
                    has_rewards = False
                else:
                    continue
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                if has_rewards:
                    if rewards.dtype != torch.float32:
                        rewards = rewards.float()
                    rewards = rewards.to(device)

                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(model, 'include_reward_head') and model.include_reward_head:
                    class_outputs, reward_outputs = model(inputs)
                else:
                    class_outputs = model(inputs)
                    reward_outputs = None
                
                # Compute losses
                class_loss = classification_criterion(class_outputs, labels)
                total_loss = class_loss
                
                if use_reward_loss and reward_outputs is not None and has_rewards:
                    reward_loss = reward_criterion(reward_outputs.squeeze(), rewards)
                    total_loss += reward_weight * reward_loss
                    running_reward_loss += reward_loss.item()
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                running_loss += total_loss.item()
                running_class_loss += class_loss.item()
                
                _, predicted = torch.max(class_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print progress
                if (i + 1) % 50 == 0:
                    print(f"  Batch {i+1}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
            
            # Validation phase
            val_loss, val_accuracy = bt_evaluate_classifier(model, val_loader, device, 
                                                       use_reward_loss, reward_weight)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Statistics
            train_accuracy = 100 * correct / total
            avg_train_loss = running_loss / len(train_loader)
            avg_class_loss = running_class_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Update the plot - THIS IS THE KEY PART
            plotter.update_plot(epoch + 1, train_accuracy, val_accuracy, avg_train_loss, val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
            print(f"  Class Loss: {avg_class_loss:.4f}")
            
            if use_reward_loss and running_reward_loss > 0:
                avg_reward_loss = running_reward_loss / len(train_loader)
                print(f"  Reward Loss: {avg_reward_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'train_accuracy': train_accuracy,
                }, 'best_barlow_classifier.pth')
                print(f"  Saved best model (Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%)")
            
            print("-" * 60)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        if not use_jupyter:
            print("Training completed. Close the plot window to continue...")
            input("Press Enter to close the plot and continue...")
        
        # Save final plot
        # plotter.save_plot('final_classifier_training_progress.png')
        plotter.close()
    
    print(f"\nTraining Summary:")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    
    return train_losses, val_losses

def height_to_pil(height_array):
    """Convert AFM height array to PIL RGB image"""
    h_min, h_max = np.min(height_array), np.max(height_array)
    norm_img = 255 * (height_array - h_min) / (h_max - h_min + 1e-8)
    norm_img = norm_img.astype(np.uint8)
    pil_img = Image.fromarray(norm_img)
    return pil_img.convert("RGB")

def extract_scan_index_from_filename(file_path):
    """Extract scan index from filename"""
    filename = os.path.basename(file_path)
    match = re.search(r'_(\d{4})', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def create_3x3_crops(pil_img, crop_size=75):
    """Create 3x3 crops from PIL image"""
    w, h = pil_img.size
    x_positions = [0, (w - crop_size) // 2, w - crop_size]
    y_positions = [0, (h - crop_size) // 2, h - crop_size]
    crops = []
    for y in y_positions:
        for x in x_positions:
            crop = pil_img.crop((x, y, x + crop_size, y + crop_size)).resize((224, 224))
            crops.append(crop)
    return crops

class BarlowTwinsClassifier(nn.Module):
    def __init__(self, encoder_path=None, num_classes=5, include_reward_head=True, dropout_rate=0.3, freeze_encoder=True
                 ):
        super(BarlowTwinsClassifier, self).__init__()
        
        # Encoder (ResNet18)
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        
        # Optionally load encoder weights
        if encoder_path is not None and os.path.exists(encoder_path):
            state_dict = torch.load(encoder_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=False)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        feature_dim = 512  # ResNet18 feature dimension
        
        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Optional reward prediction head
        self.include_reward_head = include_reward_head
        if include_reward_head:
            self.reward_predictor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        
        class_output = self.classifier(features)
        
        reward_output = None
        if self.include_reward_head:
            reward_output = self.reward_predictor(features)
        
        if return_features:
            return class_output, reward_output, features
        else:
            if self.include_reward_head:
                return class_output, reward_output
            else:
                return class_output

    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        
        class_output = self.classifier(features)
        
        reward_output = None
        if self.include_reward_head:
            reward_output = self.reward_predictor(features)
        
        if return_features:
            return class_output, reward_output, features
        else:
            if self.include_reward_head:
                return class_output, reward_output
            else:
                return class_output

def bt_load_multiclass_classifier(model_path, device='cpu', num_classes=5):
    """Load the trained multiclass classifier"""
    checkpoint = torch.load(model_path, map_location=device)
    has_reward_head = any(k.startswith("reward_predictor") for k in checkpoint['model_state_dict'].keys())
    
    # Note this makes the loader non-configurable, auto-done
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch, 'has_mps') and torch.has_mps and torch.mps.is_available() else 'cpu')

    model = BarlowTwinsClassifier(
        num_classes=num_classes, 
        include_reward_head=has_reward_head,
        dropout_rate=0.3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return model

def bt_predict_with_probabilities(model, transform, ibw_file_path, device='cpu'):
    """Predict on a single IBW file and return class probabilities"""
    try:
        # Load IBW file
        ibw_data = load_ibw(ibw_file_path)
        height_img = ibw_data.z
        
        scan_index = extract_scan_index_from_filename(ibw_file_path)
        
        # Convert to PIL and create crops
        pil_img = height_to_pil(height_img)
        crops = create_3x3_crops(pil_img)
        crop_tensors = torch.stack([transform(c) for c in crops], dim=0).to(device)

        # Make predictions
        with torch.no_grad():
            if model.include_reward_head:
                outputs, _ = model(crop_tensors)
            else:
                outputs = model(crop_tensors)
                
            probs = torch.softmax(outputs, dim=1)
            avg_probs = probs.mean(dim=0)  # Average across all crops

        return {
            "file": ibw_file_path,
            "probabilities": avg_probs.cpu().numpy(),
            "scan_index": scan_index,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "file": ibw_file_path,
            "probabilities": None,
            "scan_index": extract_scan_index_from_filename(ibw_file_path),
            "status": "error",
            "error": str(e)
        }

def bt_generate_ground_truth_labels(files_data, subfolder_name, num_classes=5):
    """Generate ground truth labels based on subfolder and scan indices"""
    labels = []
    
    for data in files_data:
        scan_idx = data["scan_index"]
        
        if subfolder_name == "read_out":
            # Early scans should be fresh tips (class 0-1)
            if scan_idx <= 10:
                gt_label = 0  # Fresh Tip
            elif scan_idx <= 20:
                gt_label = 1  # Slight Wear
            else:
                gt_label = 2  # Moderate Wear
        else:  # wear_out
            # Later scans should be more worn (class 2-4)
            if scan_idx <= 10:
                gt_label = 2  # Moderate Wear
            elif scan_idx <= 20:
                gt_label = 3  # Significant Wear
            else:
                gt_label = 4  # Heavily Worn
        
        labels.append(gt_label)
    
    return labels

def bt_collect_prediction_data(model_path, base_folder, subfolders=None, device='cpu', num_classes=5):
    """Collect prediction probabilities and ground truth labels for ROC analysis"""
    
    from tqdm import tqdm

    if subfolders is None:
        subfolders = ["read_out", "wear_out"]
    
    print(f"Collecting prediction data for ROC analysis...")
    print(f"Using device: {device}")
    
    # Load model
    model = bt_load_multiclass_classifier(model_path, device, num_classes)
    
    # Setup transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_probabilities = []
    all_labels = []
    all_file_info = []
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue
            
        print(f"Processing {subfolder} folder...")
        
        # Get IBW files
        ibw_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                    if f.lower().endswith('.ibw')]
        
        if not ibw_files:
            print(f"No IBW files found in {folder_path}")
            continue
        
        # Collect predictions
        subfolder_data = []
        for ibw_file in tqdm(ibw_files, desc=f"Processing {subfolder}"):
            result = bt_predict_with_probabilities(model, transform, ibw_file, device)
            if result["status"] == "success":
                subfolder_data.append(result)
        
        # Generate ground truth labels for this subfolder
        gt_labels = bt_generate_ground_truth_labels(subfolder_data, subfolder, num_classes)
        
        # Collect data
        for i, data in enumerate(subfolder_data):
            all_probabilities.append(data["probabilities"])
            all_labels.append(gt_labels[i])
            all_file_info.append({
                "file": data["file"],
                "subfolder": subfolder,
                "scan_index": data["scan_index"],
                "ground_truth": gt_labels[i]
            })
    
    return np.array(all_probabilities), np.array(all_labels), all_file_info

def bt_plot_multiclass_roc_curves(y_true, y_probs, num_classes=5, output_dir="roc_analysis"):
    """Generate and save ROC curves for multiclass classification"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Class labels
    class_names = [
        "Fresh Tip",
        "Slight Wear", 
        "Moderate Wear",
        "Significant Wear",
        "Heavily Worn"
    ]
    
    # Binarize the output labels for multiclass ROC
    y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_probs.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # # Compute macro-average ROC curve and ROC area
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(num_classes):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # mean_tpr /= num_classes
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 8))
    
    # Colors for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    
    # Plot individual class ROC curves
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # # Plot micro and macro averages
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
    #          color='deeppink', linestyle=':', linewidth=4)
    
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
    #          color='navy', linestyle=':', linewidth=4)
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Tool Wear Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig(os.path.join(output_dir, f'multiclass_roc_curves_{timestamp}.png'), 
    #             dpi=300, bbox_inches='tight')
    plt.show()
    
    # # Create a summary dataframe
    # roc_summary = pd.DataFrame({
    #     'Class': class_names + ['Micro-Average', 'Macro-Average'],
    #     'AUC': [roc_auc[i] for i in range(num_classes)] + [roc_auc['micro'], roc_auc['macro']]
    # })

    # Create a summary dataframe
    roc_summary = pd.DataFrame({
        'Class': class_names,
        'AUC': [roc_auc[i] for i in range(num_classes)]
    })
    
    # Save summary
    # roc_summary.to_csv(os.path.join(output_dir, f'roc_auc_summary_{timestamp}.csv'), index=False)
    
    return roc_auc, roc_summary

def bt_plot_individual_class_roc(y_true, y_probs, num_classes=5, output_dir="roc_analysis"):
    """Plot individual ROC curves for each class"""
    
    class_names = [
        "Fresh Tip",
        "Slight Wear", 
        "Moderate Wear",
        "Significant Wear",
        "Heavily Worn"
    ]
    
    # Binarize the output labels
    y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        axes[i].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{class_names[i]}')
        axes[i].legend(loc="lower right")
        axes[i].grid(True, alpha=0.3)
    
    # Hide the last subplot if we have an odd number of classes
    if num_classes < 6:
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig(os.path.join(output_dir, f'individual_roc_curves_{timestamp}.png'), 
    #             dpi=300, bbox_inches='tight')
    plt.show()

def bt_generate_confusion_matrix_heatmap(y_true, y_pred, output_dir="roc_analysis"):
    """Generate a confusion matrix heatmap"""
    
    class_names = [
        "Fresh Tip",
        "Slight Wear", 
        "Moderate Wear",
        "Significant Wear",
        "Heavily Worn"
    ]
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Tool Wear Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig(os.path.join(output_dir, f'confusion_matrix_{timestamp}.png'), 
    #             dpi=300, bbox_inches='tight')
    plt.show()

def bt_run_complete_roc_analysis(model_path, base_folder, subfolders=None, 
                             output_dir="roc_analysis", device='cpu', num_classes=5):
    """Run complete ROC analysis including data collection and visualization"""
    
    print("=== STARTING ROC ANALYSIS ===")
    
    # Collect prediction data
    y_probs, y_true, file_info = bt_collect_prediction_data(
        model_path, base_folder, subfolders, device, num_classes
    )
    
    if len(y_true) == 0:
        print("No valid prediction data found. Cannot generate ROC curves.")
        return
    
    print(f"Collected {len(y_true)} predictions for ROC analysis")
    
    # Generate predicted classes
    y_pred = np.argmax(y_probs, axis=1)
    
    # Calculate overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"Overall Accuracy: {accuracy:.3f}")
    
    # Generate ROC curves
    print("Generating ROC curves...")
    roc_auc, roc_summary = bt_plot_multiclass_roc_curves(y_true, y_probs, num_classes, output_dir)
    
    # Generate individual class ROC curves
    print("Generating individual class ROC curves...")
    bt_plot_individual_class_roc(y_true, y_probs, num_classes, output_dir)
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    bt_generate_confusion_matrix_heatmap(y_true, y_pred, output_dir)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create detailed results dataframe
    detailed_results = []
    for i, info in enumerate(file_info):
        result = {
            'file': info['file'],
            'subfolder': info['subfolder'],
            'scan_index': info['scan_index'],
            'ground_truth': y_true[i],
            'predicted': y_pred[i],
            'correct': y_true[i] == y_pred[i],
            'confidence': y_probs[i][y_pred[i]]
        }
        # Add class probabilities
        for j in range(num_classes):
            result[f'prob_class_{j}'] = y_probs[i][j]
        detailed_results.append(result)
    
    results_df = pd.DataFrame(detailed_results)
    # results_df.to_csv(os.path.join(output_dir, f'detailed_roc_results_{timestamp}.csv'), index=False)
    
    print(f"\n=== ROC ANALYSIS SUMMARY ===")
    print(roc_summary)
    # print(f"\nResults saved to: {output_dir}")
    
    return roc_auc, roc_summary, results_df

def bt_plot_classification_vs_scan_index(results, title="Classification vs Scan Index with Trendlines"):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression

    scan_indices = []
    predicted_classes = []
    confidences = []
    categories = []

    for r in results:
        if 'scan_index' in r and r['scan_index'] is not None and r['predicted_class'] is not None:
            scan_indices.append(r['scan_index'])
            predicted_classes.append(r['predicted_class'])
            confidences.append(r['confidence'])
            if 'read_out' in r['file'].lower():
                categories.append('read_out')
            elif 'wear_out' in r['file'].lower():
                categories.append('wear_out')
            else:
                categories.append('unknown')

    scan_indices = np.array(scan_indices)
    predicted_classes = np.array(predicted_classes)
    confidences = np.array(confidences)
    categories = np.array(categories)

    plt.figure(figsize=(10, 6))

    # Plot read_out
    read_mask = categories == 'read_out'
    plt.scatter(scan_indices[read_mask], predicted_classes[read_mask],
                c=confidences[read_mask], cmap='viridis', marker='o', s=50, label='Read Out')

    # Plot wear_out
    wear_mask = categories == 'wear_out'
    plt.scatter(scan_indices[wear_mask], predicted_classes[wear_mask],
                c=confidences[wear_mask], cmap='viridis', marker='s', s=50, label='Wear Out')

    # Trendlines
    for mask, label, color in zip([read_mask, wear_mask], ['Read Out', 'Wear Out'], ['blue', 'red']):
        if np.sum(mask) > 1:
            X = scan_indices[mask].reshape(-1, 1)
            y = predicted_classes[mask]
            model = LinearRegression()
            model.fit(X, y)
            x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            plt.plot(x_line, y_line, color=color, linestyle='--', label=f'{label} Trend')

    plt.colorbar(label="Confidence")
    plt.xlabel("Scan Index")
    plt.ylabel("Predicted Class")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def bt_plot_confidence_vs_predicted_class(results, title="Confidence vs Predicted Class"):
    import matplotlib.pyplot as plt
    import numpy as np

    predicted_classes = []
    confidences = []
    categories = []

    for r in results:
        if r['predicted_class'] is not None and r['confidence'] is not None:
            # Ensure predicted_class is integer (discrete)
            predicted_classes.append(int(r['predicted_class']))
            # Optionally, round confidence to nearest 2 decimals for x-axis clarity, but keep as float
            confidences.append(r['confidence'])
            if 'read_out' in r['file'].lower():
                categories.append('read_out')
            elif 'wear_out' in r['file'].lower():
                categories.append('wear_out')
            else:
                categories.append('unknown')

    predicted_classes = np.array(predicted_classes)
    confidences = np.array(confidences)
    categories = np.array(categories)

    plt.figure(figsize=(10, 6))

    read_mask = categories == 'read_out'
    wear_mask = categories == 'wear_out'

    plt.scatter(confidences[read_mask], predicted_classes[read_mask], color='blue', marker='o', label='Read Out', alpha=0.7)
    plt.scatter(confidences[wear_mask], predicted_classes[wear_mask], color='red', marker='s', label='Wear Out', alpha=0.7)

    plt.xlabel("Confidence")
    plt.ylabel("Predicted Class")
    plt.title(title)
    plt.yticks(sorted(set(predicted_classes)))  # Only show integer class ticks
    plt.legend()
    plt.grid(True)
    plt.show()


"""
PCA Analysis for comparsion of the two mode.
"""

class PCAAnalyzer:
    """
    A comprehensive PCA analysis tool for comparing AFM tip classification models.
    Designed to work with RewardAwareModel and Barlow Twins models.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.features = {}
        self.labels = {}
        self.pca_results = {}
        self.quality_descriptions = {
            0: "Excellent (Class 0)",
            1: "Good (Class 1)", 
            2: "Fair (Class 2)",
            3: "Poor (Class 3)",
            4: "Bad (Class 4)"
        }
        
    def load_hybrid_model(self, model_path: str, model_name: str = "Hybrid"):
        """Load the Hybrid RewardAwareModel"""
        from tools import RewardAwareModel
        
        model = RewardAwareModel(num_classes=5, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        print(f"Loaded {model_name} model from {model_path}")
        
    def load_barlow_twins_model(self, model_path: str, model_name: str = "Barlow_Twins", num_classes: int = 5):
        """
        Load the Barlow Twins model using the provided BarlowTwinsClassifier.
        
        Args:
            model_path: Path to the saved model
            model_name: Name to use for this model
            num_classes: Number of classes for the classifier
        """
        # Load checkpoint to check for reward head
        checkpoint = torch.load(model_path, map_location=self.device)
        has_reward_head = any(k.startswith("reward_predictor") for k in checkpoint['model_state_dict'].keys())
        
        # Create model with appropriate configuration
        model = BarlowTwinsClassifier(
            num_classes=num_classes, 
            include_reward_head=has_reward_head,
            dropout_rate=0.3  # Using lower dropout for inference
        )
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        print(f"Loaded {model_name} model from {model_path}")
        print(f"  - Has reward head: {has_reward_head}")
        print(f"  - Feature dimension: 512")
        
    def extract_features_from_dataloader(self, model_name: str, dataloader, 
                                       max_samples: Optional[int] = None):
        """
        Extract final layer features from a dataloader
        
        Args:
            model_name: Name of the model to use
            dataloader: PyTorch DataLoader with your test data
            max_samples: Maximum number of samples to process (None for all)
        """
        model = self.models[model_name]
        model.eval()
        
        all_features = []
        all_labels = []
        sample_count = 0
        
        print(f"Extracting features using {model_name} model...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Handle your custom collate function output
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['label']
                else:
                    images, labels = batch[0].to(self.device), batch[1]
                
                # Extract features
                if hasattr(model, 'return_features') or 'return_features' in model.forward.__code__.co_varnames:
                    # For both RewardAwareModel and BarlowTwinsClassifier
                    if hasattr(model, 'include_reward_head') and model.include_reward_head:
                        # Barlow Twins with reward head
                        _, _, features = model(images, return_features=True)
                    else:
                        # Barlow Twins without reward head or Hybrid model
                        try:
                            _, _, features = model(images, return_features=True)
                        except ValueError:
                            # Handle case where model returns only class output
                            class_output, features = model(images, return_features=True)
                else:
                    # Fallback for other models
                    features = model.get_features(images)
                
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
                
                sample_count += len(labels)
                if max_samples and sample_count >= max_samples:
                    break
                    
                if batch_idx % 10 == 0:
                    print(f"  Processed {sample_count} samples...")
        
        features_array = np.vstack(all_features)
        labels_array = np.array(all_labels)
        
        self.features[model_name] = features_array
        self.labels[model_name] = labels_array
        
        print(f"Extracted {len(features_array)} feature vectors of dimension {features_array.shape[1]}")
        
    def extract_features_from_files(self, model_name: str, ibw_files: List[str], 
                                  transform, max_samples: Optional[int] = None):
        """
        Extract features from individual IBW files
        
        Args:
            model_name: Name of the model to use
            ibw_files: List of IBW file paths
            transform: Transform to apply to images
            max_samples: Maximum number of files to process
        """
        from tools import load_ibw  # Assuming this import works
        from PIL import Image
        
        model = self.models[model_name]
        model.eval()
        
        all_features = []
        all_labels = []
        
        files_to_process = ibw_files[:max_samples] if max_samples else ibw_files
        print(f"Extracting features from {len(files_to_process)} IBW files using {model_name}...")
        
        with torch.no_grad():
            for i, file_path in enumerate(files_to_process):
                try:
                    # Load IBW file
                    ibw_data = load_ibw(file_path)
                    height_img = ibw_data.z
                    
                    # Convert to PIL image
                    h_min, h_max = np.min(height_img), np.max(height_img)
                    norm_img = 255 * (height_img - h_min) / (h_max - h_min + 1e-8)
                    norm_img = norm_img.astype(np.uint8)
                    pil_img = Image.fromarray(norm_img).convert("RGB")
                    
                    # Apply transform
                    img_tensor = transform(pil_img).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    if hasattr(model, 'return_features') or 'return_features' in model.forward.__code__.co_varnames:
                        # For both RewardAwareModel and BarlowTwinsClassifier
                        if hasattr(model, 'include_reward_head') and model.include_reward_head:
                            # Barlow Twins with reward head
                            _, _, features = model(img_tensor, return_features=True)
                        else:
                            # Barlow Twins without reward head or Hybrid model
                            try:
                                _, _, features = model(img_tensor, return_features=True)
                            except ValueError:
                                # Handle case where model returns only class output
                                class_output, features = model(img_tensor, return_features=True)
                    else:
                        # Fallback
                        features = model.get_features(img_tensor)
                    
                    all_features.append(features.cpu().numpy())
                    
                    # Extract label from filename (you might need to adjust this)
                    label = self._extract_label_from_filename(file_path)
                    all_labels.append(label)
                    
                    if i % 50 == 0:
                        print(f"  Processed {i+1}/{len(files_to_process)} files...")
                        
                except Exception as e:
                    print(f"  Error processing {file_path}: {e}")
                    continue
        
        if all_features:
            features_array = np.vstack(all_features)
            labels_array = np.array(all_labels)
            
            self.features[model_name] = features_array
            self.labels[model_name] = labels_array
            
            print(f"Extracted {len(features_array)} feature vectors of dimension {features_array.shape[1]}")
        else:
            print("No features extracted!")
            
    def _extract_label_from_filename(self, filename: str) -> int:
        """
        Extract class label from filename. Adjust this based on your naming convention.
        This is a placeholder - you'll need to implement based on your file naming.
        """
        # Example: if files are named like "class_0_sample_001.ibw"
        # You'll need to adjust this based on your actual file naming convention
        import re
        match = re.search(r'class_(\d+)', filename.lower())
        if match:
            return int(match.group(1))
        else:
            # Default to class 0 if can't extract
            print(f"Warning: Could not extract label from {filename}, defaulting to class 0")
            return 0
    
    def perform_pca(self, model_name: str, n_components: int = 10, standardize: bool = True):
        """
        Perform PCA analysis on extracted features
        
        Args:
            model_name: Name of the model
            n_components: Number of principal components
            standardize: Whether to standardize features before PCA
        """
        if model_name not in self.features:
            raise ValueError(f"No features found for model {model_name}. Extract features first.")
        
        features = self.features[model_name]
        
        # Standardize features if requested
        if standardize:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        else:
            features_scaled = features
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)
        
        # Store results
        self.pca_results[model_name] = {
            'pca_object': pca,
            'features_pca': features_pca,
            'features_scaled': features_scaled,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'n_components': n_components
        }
        
        print(f"PCA completed for {model_name}")
        print(f"  First 5 components explain {pca.explained_variance_ratio_[:5].sum():.3f} of variance")
        
    def plot_variance_explained(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot explained variance for all models"""
        if not self.pca_results:
            print("No PCA results found. Run perform_pca() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance explained
        for model_name, results in self.pca_results.items():
            n_comp = min(10, len(results['explained_variance_ratio']))
            ax1.bar(range(1, n_comp + 1), results['explained_variance_ratio'][:n_comp], 
                   alpha=0.7, label=model_name)
        
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained')
        ax1.set_title('Variance Explained by Each Component')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance explained
        for model_name, results in self.pca_results.items():
            n_comp = min(10, len(results['cumulative_variance']))
            ax2.plot(range(1, n_comp + 1), results['cumulative_variance'][:n_comp], 
                    'o-', label=model_name, linewidth=2)
        
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95%')
        
        plt.tight_layout()
        plt.show()
        
    def plot_pca_scatter(self, pc1: int = 0, pc2: int = 1, figsize: Tuple[int, int] = (15, 6)):
        """
        Create scatter plots of PCA results for all models
        
        Args:
            pc1, pc2: Which principal components to plot (0-indexed)
            figsize: Figure size
        """
        if not self.pca_results:
            print("No PCA results found. Run perform_pca() first.")
            return
        
        n_models = len(self.pca_results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, 5))  # 5 classes
        
        for idx, (model_name, results) in enumerate(self.pca_results.items()):
            ax = axes[idx]
            features_pca = results['features_pca']
            labels = self.labels[model_name]
            
            # Plot each class
            for class_idx in range(5):
                mask = labels == class_idx
                if np.any(mask):
                    ax.scatter(features_pca[mask, pc1], features_pca[mask, pc2], 
                             c=[colors[class_idx]], alpha=0.6, s=30,
                             label=self.quality_descriptions[class_idx])
            
            ax.set_xlabel(f'PC{pc1+1} ({results["explained_variance_ratio"][pc1]:.2%} variance)')
            ax.set_ylabel(f'PC{pc2+1} ({results["explained_variance_ratio"][pc2]:.2%} variance)')
            ax.set_title(f'{model_name} - PCA Feature Space')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_class_separation(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Analyze class separation in PCA space
        """
        if not self.pca_results:
            print("No PCA results found. Run perform_pca() first.")
            return
        
        n_models = len(self.pca_results)
        fig, axes = plt.subplots(2, n_models, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (model_name, results) in enumerate(self.pca_results.items()):
            features_pca = results['features_pca']
            labels = self.labels[model_name]
            
            # Plot 1: Distribution of first PC by class
            ax1 = axes[0, idx]
            for class_idx in range(5):
                mask = labels == class_idx
                if np.any(mask):
                    ax1.hist(features_pca[mask, 0], alpha=0.6, bins=20, 
                            label=f'Class {class_idx}', density=True)
            
            ax1.set_xlabel('First Principal Component')
            ax1.set_ylabel('Density')
            ax1.set_title(f'{model_name} - PC1 Distribution by Class')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Distance from origin by class
            ax2 = axes[1, idx]
            distances = np.sqrt(np.sum(features_pca[:, :2]**2, axis=1))
            
            class_distances = []
            class_labels = []
            for class_idx in range(5):
                mask = labels == class_idx
                if np.any(mask):
                    class_distances.extend(distances[mask])
                    class_labels.extend([f'Class {class_idx}'] * np.sum(mask))
            
            # Box plot of distances
            import pandas as pd
            df = pd.DataFrame({'Distance': class_distances, 'Class': class_labels})
            sns.boxplot(data=df, x='Class', y='Distance', ax=ax2)
            ax2.set_title(f'{model_name} - Distance from Origin (PC1-PC2)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def compare_models(self):
        """
        Generate a comprehensive comparison between models
        """
        if len(self.pca_results) < 2:
            print("Need at least 2 models for comparison")
            return
        
        print("=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        for model_name, results in self.pca_results.items():
            print(f"\n{model_name} Model:")
            print(f"  Feature dimension: {self.features[model_name].shape[1]}")
            print(f"  Number of samples: {len(self.features[model_name])}")
            print(f"  PC1 explains: {results['explained_variance_ratio'][0]:.2%} of variance")
            print(f"  PC1+PC2 explains: {results['explained_variance_ratio'][:2].sum():.2%} of variance")
            print(f"  First 5 PCs explain: {results['explained_variance_ratio'][:5].sum():.2%} of variance")
            
            # Class distribution
            unique, counts = np.unique(self.labels[model_name], return_counts=True)
            print(f"  Class distribution: {dict(zip(unique, counts))}")
        
        print("\n" + "=" * 60)
        
    def save_results(self, save_path: str):
        """Save PCA results to file"""
        results_to_save = {
            'features': self.features,
            'labels': self.labels,
            'pca_results': {name: {k: v for k, v in results.items() if k != 'pca_object'} 
                           for name, results in self.pca_results.items()}
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print(f"Results saved to {save_path}")

def run_pca_analysis(hybrid_model_path: str, barlow_model_path: str, 
                    test_dataloader, device='cpu', max_samples=1000):
    """
    Complete PCA analysis workflow
    
    Args:
        hybrid_model_path: Path to saved hybrid model
        barlow_model_path: Path to saved Barlow Twins model
        test_dataloader: DataLoader with test data
        device: PyTorch device
    """

    # Initialize analyzer
    analyzer = PCAAnalyzer(device=device)
    
    # Load models
    analyzer.load_hybrid_model(hybrid_model_path, "Hybrid")
    analyzer.load_barlow_twins_model(barlow_model_path, "Barlow_Twins")
    
    # Extract features from both models
    analyzer.extract_features_from_dataloader("Hybrid", test_dataloader, max_samples=max_samples)
    analyzer.extract_features_from_dataloader("Barlow_Twins", test_dataloader, max_samples=max_samples)
    
    # Perform PCA
    analyzer.perform_pca("Hybrid", n_components=10)
    analyzer.perform_pca("Barlow_Twins", n_components=10)
    
    # Generate visualizations
    analyzer.plot_variance_explained()
    analyzer.plot_pca_scatter(pc1=0, pc2=1)
    analyzer.plot_pca_scatter(pc1=1, pc2=2)
    analyzer.plot_class_separation()
    
    # Print comparison
    analyzer.compare_models()
    
    # Save results
    analyzer.save_results("pca_analysis_results.pkl")
    
    return analyzer

"""
Combination of Barlow Twins model and Hybrid Model
"""

class CombinedBT_HB_Classifier:
    def __init__(self, bt_model_path, hb_model_path, parent_folder):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch, 'has_mps') and torch.has_mps and torch.mps.is_available() else 'cpu')


        # Intialize both a BT and HB Classifier from loaded file (training takes too long)
        self.bt_model = bt_load_multiclass_classifier(bt_model_path, device=self.device)
        self.hb_model = load_trained_hybrid_model(hb_model_path, device=self.device)

        self.model_paths = {
            "barlow_twins": bt_model_path,
            "hybrid": hb_model_path
        }

        # Pre-defined, same as training data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.results = {
            "bt_result": [],
            "hb_result": []
        }

        self.data = {
            "scan_indices": [],
            "class_label": [],
            "disagreed_classes": [],
            "average_class_difference": 0,
            "most_common_class_disagreement": 0
        }

        file_paths = []
        file_paths = get_all_ibw_files(parent_folder)

        for i, file_path in enumerate(file_paths):
            print(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
        
            # Get scan index if available from training
            scan_index = extract_scan_index_from_filename(file_path)
            self.data["scan_indices"].append(scan_index)

        # Map scan_indices to class labels
        if self.data["scan_indices"]:
            self.data["class_label"] = generate_labels_from_scan_indices(self.data["scan_indices"])
    
        file_paths = []
        file_paths = get_all_ibw_files(parent_folder)

        for i, file_path in enumerate(file_paths):
            print(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
        
            # Get scan index if available from training
            self.data["scan_indices"].append(extract_scan_index_from_filename(file_path))

        # No further instantiations needed
    
    def dual_predict(self, file_path):

        self.results['bt_result'] = bt_predict_with_probabilities(self.bt_model,
                                                              self.transform, 
                                                              file_path, 
                                                              device=self.device)
    
        bt_predicted_class = np.argmax(self.results['bt_result']['probabilities'])

        if 'error' not in self.results['bt_result']['status']:
            print(f"Barlow Twins Prediction: {bt_predicted_class} with confidence {self.results['bt_result']['probabilities']} ")
        else:
            print(f"   Error: {self.results['bt_result']['status']}, check configuration.")


        # Define the Augmented Transform (same as training)
        aug_transform = AugmentedTransform(base_size=256, crop_size=86, normalize=True)

        # If possible, grab the scan index from the filename (normalized)
        # (If this isn't available, it will just default to 0)
        scan_index = extract_scan_index_from_filename(file_path)

        self.results['hb_result'] = hb_predict_ibw(self.hb_model,
                                                   file_path, 
                                                   aug_transform,
                                                   self.device,
                                                   scan_index=scan_index)

        if 'error' not in self.results['hb_result']:
            print(f"Scan Index: {scan_index} Predicted: Class {self.results['hb_result']['predicted_class']} ({self.results['hb_result']['quality_description']}) "
                  f"with {self.results['hb_result']['confidence']:.3f} confidence")
        else:
            print(f"  Error: {self.results['hb_result']['error']}")

        # We now have access to both model's results, which means that we can join
        # their combined prediction.

        # Call to clear the console
        # clear_console()
        print('\n \n')

        # For use in CLI
        BOLD_START = '\033[1m'
        BOLD_END = '\033[0m'

        # Start comparison for errors
        if np.argmax(self.results['bt_result']['probabilities']) != self.results['hb_result']['predicted_class']:
            print(f"Models {BOLD_START}did not agree{BOLD_END} on classification of {file_path}.")
            print(f"Hybrid model classified input as class {BOLD_START}{self.results['hb_result']['predicted_class']}{BOLD_END}.")
            print(f"Barlow Twins model classified input as class {BOLD_START}{np.argmax(self.results['bt_result']['probabilities'])}{BOLD_END}")

            # Add the disagreed classes (not from indices) here
            # Find the scan index for this file
            scan_index = extract_scan_index_from_filename(file_path)
            # Find the true class label for this scan index (if available)
            if scan_index in self.data["scan_indices"]:
                idx = self.data["scan_indices"].index(scan_index)
                true_class = self.data["class_label"][idx]
            else:
                true_class = None

            # Store the disagreement info with true class label
            self.data['disagreed_classes'].append(true_class)
            print(f"True class label for scan index {scan_index}: {true_class}")

        else:
            print(f"Models {BOLD_START}agreed{BOLD_END} on classification of {file_path} as {BOLD_START}class {self.results['hb_result']['predicted_class']}{BOLD_END}.")

    def process_disagreement(self):
        # For use in CLI
        BOLD_START = '\033[1m'
        BOLD_END = '\033[0m'

        print(f"Disagreed classes (list): {self.data['disagreed_classes']}")
        self.data["average_class_difference"] = np.average(self.data['disagreed_classes'])
        
        def most_frequent(List):
            return max(set(List), key=List.count)
        
        self.data["most_common_class_disagreement"] = most_frequent(self.data["disagreed_classes"])
        
        print(f"Average class disagreed on: {BOLD_START}Class {int(round(self.data['average_class_difference']))}{BOLD_END}")
        print(f"Most common class disagreed on: {BOLD_START}Class {self.data["most_common_class_disagreement"]}{BOLD_END}")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Run dual prediction using CombinedBT_HB_Classifier.")
#     parser.add_argument('--file', type=str, required=False, nargs='+',
#                         default=['exp_data/June 18/wear_out/Wear_out_0015.ibw'],
#                         help='Path to the .ibw file to classify ')
#     args = parser.parse_args()

#     # Join the file path parts in case of spaces
#     file_path = ' '.join(args.file)
#     print(f"{file_path}")

#     parent_folder = 'exp_data/June 18/'  # file_path is the parent folder path from args
#     file_paths = get_all_ibw_files(parent_folder)
#     print(f"Found {len(file_paths)} .ibw files in {parent_folder} and its subfolders.")

#     bt_model_path = 'barlow_twins_multiclass_classifier_v1.pth'
#     hb_model_path = 'hybrid_model_v1.pth'
#     cmodel = CombinedBT_HB_Classifier(bt_model_path, hb_model_path, parent_folder)

#     for file_path in file_paths:
#         cmodel.dual_predict(file_path)
#         cmodel.process_disagreement()
