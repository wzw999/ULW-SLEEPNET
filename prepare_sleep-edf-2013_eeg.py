import os
import glob
import ntpath
import logging
import argparse

import pyedflib
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal.windows import hamming


# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
MOVE = 5
UNK = 6

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK
}

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def split_segments(data, labels, segment_length, target_length, fs):
    """
    Split data and labels into smaller segments, and apply Hamming window to each segment.
    :param data: np.ndarray, shape (num_segments, channels, samples_per_segment)
    :param labels: np.ndarray, shape (num_segments, num_classes)
    :param segment_length: int, original segment length in seconds
    :param target_length: int, target segment length in seconds
    :param fs: int, sampling frequency
    :return: split_data, split_labels
    """
    samples_per_segment = segment_length * fs
    samples_per_target = target_length * fs
    num_targets = samples_per_segment // samples_per_target

    split_data = []
    split_labels = []
    
    # 创建汉明窗
    hamming_window = hamming(samples_per_target)

    for i in range(data.shape[0]):
        for j in range(num_targets):
            start = j * samples_per_target
            end = start + samples_per_target
            segment = data[i, :, start:end]
            
            # 对每个通道应用汉明窗
            for ch in range(segment.shape[0]):
                segment[ch, :] = segment[ch, :] * hamming_window
                
            split_data.append(segment)
            split_labels.append(labels[i])

    return np.array(split_data), np.array(split_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./edf",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="./npz",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    parser.add_argument("--segment_length", type=int, default=30,
                        help="Original segment length in seconds.")
    parser.add_argument("--target_length", type=int, default=30,
                        help="Target segment length in seconds for splitting.")
    args = parser.parse_args()
    
    # Select channels - only keep these 4 channels
    # select_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
    select_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    
    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    # Initialize data storage for all subjects
    fold_data = []
    fold_label = []
    fold_len = []
    all_found_channels = []

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = psg_f.datarecord_duration
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signals from the selected channels
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()
        
        # Find indices for all selected channels
        select_ch_indices = []
        found_channels = []
        for select_ch in select_channels:
            select_ch_idx = -1
            for s in range(psg_f.signals_in_file):
                if ch_names[s] == select_ch:
                    select_ch_idx = s
                    break
            if select_ch_idx != -1:
                select_ch_indices.append(select_ch_idx)
                found_channels.append(select_ch)
            else:
                logger.warning(f"Channel {select_ch} not found in file {psg_fnames[i]}")
        
        if len(select_ch_indices) == 0:
            logger.error(f"No selected channels found in file {psg_fnames[i]}")
            continue
        
        # Store found channels for the first file (assuming all files have same channels)
        if i == 0:
            all_found_channels = found_channels
        
        sampling_rate = 100.0
        n_epoch_samples = int(epoch_duration * sampling_rate)
        
        # Extract signals from all selected channels
        signals_list = []
        actual_sampling_rates = []
        
        for idx in select_ch_indices:
            # Get the actual sampling rate for this channel
            signal_header = psg_f.getSignalHeader(idx)
            
            logger.debug(f"Signal header type: {type(signal_header)}")
            logger.debug(f"Signal header content: {signal_header}")
            
            # Try different possible keys for sampling rate
            actual_sr = None
            possible_sr_keys = ['sample_rate', 'sample_frequency', 'sampling_rate', 'fs']
            
            # Handle case where signal_header might not be a dict
            if hasattr(signal_header, 'keys'):
                for key in possible_sr_keys:
                    if key in signal_header:
                        actual_sr = signal_header[key]
                        break
            elif hasattr(signal_header, '__getitem__'):
                # Try direct indexing for list-like objects
                try:
                    for key in possible_sr_keys:
                        actual_sr = getattr(signal_header, key, None)
                        if actual_sr is not None:
                            break
                except AttributeError:
                    pass
            
            if actual_sr is None:
                # Fallback: try to get sampling rate from the EDF reader directly
                try:
                    actual_sr = psg_f.getSampleFrequency(idx)
                    logger.info(f"Using getSampleFrequency() fallback for channel {idx}: {actual_sr} Hz")
                except:
                    logger.error(f"Could not determine sampling rate for channel {idx}")
                    logger.error(f"Signal header type: {type(signal_header)}")
                    logger.error(f"Signal header content: {signal_header}")
                    if hasattr(signal_header, 'keys'):
                        logger.error(f"Available keys: {list(signal_header.keys())}")
                    raise KeyError(f"No sampling rate found for channel {idx}")
            
            actual_sampling_rates.append(actual_sr)
            
            logger.info(f"Channel {found_channels[len(signals_list)]}: sampling rate = {actual_sr} Hz")
            
            # Calculate samples per epoch for this specific channel
            n_samples_this_channel = int(epoch_duration * actual_sr)
            
            # Read and reshape signal data
            signal_data = psg_f.readSignal(idx)
            
            # Check if reshape is possible
            if len(signal_data) % n_samples_this_channel != 0:
                logger.warning(f"Channel {found_channels[len(signals_list)]} signal length {len(signal_data)} is not divisible by {n_samples_this_channel}")
                # Truncate to the largest multiple
                truncate_length = (len(signal_data) // n_samples_this_channel) * n_samples_this_channel
                signal_data = signal_data[:truncate_length]
            
            signal_data = signal_data.reshape(-1, n_samples_this_channel)
            
            # Resample to target sampling rate (100 Hz) if needed
            if actual_sr != sampling_rate:
                target_samples = int(epoch_duration * sampling_rate)
                resampled_data = []
                for epoch in signal_data:
                    resampled_epoch = scipy_signal.resample(epoch, target_samples)
                    resampled_data.append(resampled_epoch)
                signal_data = np.array(resampled_data)
                logger.info(f"Resampled channel {found_channels[len(signals_list)]} from {actual_sr} Hz to {sampling_rate} Hz")
            
            signals_list.append(signal_data)
        
        # Ensure all signals have the same number of epochs
        min_epochs = min(sig.shape[0] for sig in signals_list)
        signals_list = [sig[:min_epochs] for sig in signals_list]
        
        # Stack signals: shape (n_epochs, n_channels, n_samples_per_epoch)
        signals = np.stack(signals_list, axis=1)
        
        logger.info("Selected channels: {}".format(found_channels))
        logger.info("Selected channel indices: {}".format(select_ch_indices))
        logger.info("Channel sampling rates: {}".format(actual_sampling_rates))
        logger.info("Target sampling rate: {}".format(sampling_rate))
        logger.info("Signals shape: {}".format(signals.shape))

        # Sanity check
        n_epochs = psg_f.datarecords_in_file
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert signals.shape[0] == n_epochs, f"signal: {signals.shape[0]} != {n_epochs}"

        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:signals.shape[0]]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Convert labels to one-hot encoding (5 classes: W, N1, N2, N3, REM)
        y_onehot = np.eye(5)[y]
        
        # Apply segmentation if needed
        if args.segment_length != args.target_length:
            logger.info(f"Splitting segments from {args.segment_length}s to {args.target_length}s")
            logger.info(f"Data before splitting: {x.shape}, labels: {y_onehot.shape}")
            
            x_split, y_split = split_segments(x, y_onehot, args.segment_length, args.target_length, int(sampling_rate))
            
            logger.info(f"Data after splitting: {x_split.shape}, labels: {y_split.shape}")
            
            # Store split data for this subject
            fold_data.append(x_split)
            fold_label.append(y_split)
            fold_len.append(len(x_split))
            
            logger.info("Subject data shape (after splitting): {}, labels shape: {}".format(x_split.shape, y_split.shape))
        else:
            # Store original data for this subject
            fold_data.append(x)
            fold_label.append(y_onehot)
            fold_len.append(len(x))
            
            logger.info("Subject data shape (no splitting): {}, labels shape: {}".format(x.shape, y_onehot.shape))
        logger.info("\n=======================================\n")

    # Save all subjects' data in the format similar to preprocess_z.py
    if len(fold_data) > 0:
        # Create filename with segment info
        # if args.segment_length != args.target_length:
        #     filename = f"sleep_edf_processed_{args.segment_length}s_to_{args.target_length}s.npz"
        # else:
        # filename = f"sleep_edf_processed_{args.target_length}s.npz"
        filename = f"sleep_edf_processed_eeg.npz"
            
        np.savez(os.path.join(args.output_dir, filename),
                Fold_data=np.array(fold_data, dtype=object),
                Fold_label=np.array(fold_label, dtype=object), 
                Fold_len=np.array(fold_len, dtype=object),
                channels=all_found_channels,
                sampling_rate=sampling_rate,
                segment_length=args.segment_length,
                target_length=args.target_length
        )
        logger.info("Saved all subjects' data to: {}".format(os.path.join(args.output_dir, filename)))
    else:
        logger.error("No data processed successfully!")


if __name__ == "__main__":
    main()