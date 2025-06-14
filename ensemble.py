# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/' # forked by jarredou

import os
import torch
import torchaudio
import soundfile as sf
import argparse
from pathlib import Path


def stft(wave, n_fft, hop_length):
    """
    Compute STFT for stereo audio using PyTorch
    :param wave: tensor of shape (channels, length)
    :param n_fft: FFT size
    :param hop_length: hop length
    :return: complex spectrogram of shape (channels, freq_bins, time_frames)
    """
    # Use torchaudio's STFT which handles multi-channel audio efficiently
    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=wave.device),
        return_complex=True
    )
    return spec


def istft(spec, hop_length, length):
    """
    Compute inverse STFT for stereo audio using PyTorch
    :param spec: complex spectrogram of shape (channels, freq_bins, time_frames)
    :param hop_length: hop length
    :param length: target length
    :return: waveform of shape (channels, length)
    """
    # Use torchaudio's ISTFT which handles multi-channel audio efficiently
    wave = torch.istft(
        spec,
        n_fft=(spec.shape[1] - 1) * 2,
        hop_length=hop_length,
        window=torch.hann_window((spec.shape[1] - 1) * 2, device=spec.device),
        length=length
    )
    return wave


def absmax(a, *, dim):
    """Find values with maximum absolute value along dimension"""
    abs_a = torch.abs(a)
    indices = torch.argmax(abs_a, dim=dim, keepdim=True)
    return torch.gather(a, dim, indices).squeeze(dim)


def absmin(a, *, dim):
    """Find values with minimum absolute value along dimension"""
    abs_a = torch.abs(a)
    indices = torch.argmin(abs_a, dim=dim, keepdim=True)
    return torch.gather(a, dim, indices).squeeze(dim)


def lambda_max(arr, dim=None, key=None):
    """Find elements with maximum key value along dimension"""
    if key is None:
        key = lambda x: x
    key_values = key(arr)
    indices = torch.argmax(key_values, dim=dim, keepdim=True)
    return torch.gather(arr, dim, indices).squeeze(dim)


def lambda_min(arr, dim=None, key=None):
    """Find elements with minimum key value along dimension"""
    if key is None:
        key = lambda x: x
    key_values = key(arr)
    indices = torch.argmin(key_values, dim=dim, keepdim=True)
    return torch.gather(arr, dim, indices).squeeze(dim)


def match_tensor_shapes(tensor_1, tensor_2):
    """Match the time dimension of two tensors by padding or trimming"""
    target_length = tensor_2.shape[-1]
    current_length = tensor_1.shape[-1]
    
    if current_length > target_length:
        tensor_1 = tensor_1[..., :target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        tensor_1 = torch.nn.functional.pad(tensor_1, (0, padding), 'constant', 0)
    
    return tensor_1


def average_waveforms(pred_track, weights, algorithm, n_fft, hop_length):
    """
    :param pred_track: tensor of shape (num, channels, length)
    :param weights: tensor of shape (num,)
    :param algorithm: One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft
    :param n_fft: FFT size for STFT operations
    :param hop_length: hop length for STFT operations
    :return: averaged waveform in shape (channels, length)
    """
    pred_track = torch.stack(pred_track)
    final_length = pred_track.shape[-1]
    weights = torch.tensor(weights, dtype=pred_track.dtype, device=pred_track.device)

    if algorithm in ['avg_wave', 'median_wave', 'min_wave', 'max_wave']:
        # Waveform domain operations
        if algorithm == 'avg_wave':
            # Weighted average
            weighted_tracks = pred_track * weights.view(-1, 1, 1)
            result = torch.sum(weighted_tracks, dim=0) / torch.sum(weights)
        elif algorithm == 'median_wave':
            result = torch.median(pred_track, dim=0)[0]
        elif algorithm == 'min_wave':
            result = lambda_min(pred_track, dim=0, key=torch.abs)
        elif algorithm == 'max_wave':
            result = lambda_max(pred_track, dim=0, key=torch.abs)
    
    elif algorithm in ['avg_fft', 'median_fft', 'min_fft', 'max_fft']:
        # Frequency domain operations
        # Convert all tracks to spectrograms
        spec_tracks = []
        for i in range(pred_track.shape[0]):
            spec = stft(pred_track[i], n_fft, hop_length)
            spec_tracks.append(spec)
        
        spec_tracks = torch.stack(spec_tracks)
        
        if algorithm == 'avg_fft':
            # Weighted average in frequency domain
            weighted_specs = spec_tracks * weights.view(-1, 1, 1, 1)
            avg_spec = torch.sum(weighted_specs, dim=0) / torch.sum(weights)
            result = istft(avg_spec, hop_length, final_length)
        elif algorithm == 'median_fft':
            # Median in frequency domain (using magnitude and phase separately)
            median_spec = torch.median(spec_tracks, dim=0)[0]
            result = istft(median_spec, hop_length, final_length)
        elif algorithm == 'min_fft':
            min_spec = lambda_min(spec_tracks, dim=0, key=torch.abs)
            result = istft(min_spec, hop_length, final_length)
        elif algorithm == 'max_fft':
            max_spec = absmax(spec_tracks, dim=0)
            result = istft(max_spec, hop_length, final_length)
    
    return result


def save_audio(waveform, sample_rate, output_path):
    """Save audio with support for different formats and bit depths using soundfile"""
    output_path = Path(output_path)
    
    # Ensure waveform is in the right format (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Convert to CPU numpy array if on GPU
    if waveform.is_cuda:
        waveform = waveform.cpu()
    
    # Convert to numpy and transpose to (samples, channels) for soundfile
    audio_np = waveform.numpy().T
    
    # Handle different output formats
    if output_path.suffix.lower() == '.flac':
        # For FLAC, convert to 24-bit
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            print(f"Clipping detected : {max_val}")
            audio_np = audio_np / max_val.numpy()
        
        # Save as FLAC with 24-bit depth
        sf.write(str(output_path), audio_np, sample_rate, subtype='PCM_24')
    else:
        # Default to float32 for WAV and other formats
        sf.write(str(output_path), audio_np, sample_rate, subtype='FLOAT')


def ensemble_files(args):
    parser = argparse.ArgumentParser(description="Audio ensemble tool using PyTorch")
    parser.add_argument("--files", type=str, required=True, nargs='+', 
                       help="Path to all audio-files to ensemble")
    parser.add_argument("--type", type=str, default='avg_wave', 
                       help="One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft")
    parser.add_argument("--weights", type=float, nargs='+', 
                       help="Weights to create ensemble. Number of weights must be equal to number of files")
    parser.add_argument("--output", default="res.wav", type=str, 
                       help="Path to output file (supports .wav, .flac)")
    parser.add_argument("--n_fft", type=int, default=2048, 
                       help="FFT size for STFT operations (default: 2048)")
    parser.add_argument("--hop_length", type=int, default=1024, 
                       help="Hop length for STFT operations (default: 1024)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)")
    parser.add_argument("--trim_to_shortest", action="store_true", 
                       help="Trim output to shortest input file length to avoid padding artifacts")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    print(f'Ensemble type: {args.type}')
    print(f'Number of input files: {len(args.files)}')
    print(f'N_FFT: {args.n_fft}, Hop length: {args.hop_length}')
    
    if args.weights is not None:
        if len(args.weights) != len(args.files):
            raise ValueError("Number of weights must match number of files")
        weights = args.weights
    else:
        weights = [1.0] * len(args.files)
    
    print(f'Weights: {weights}')
    print(f'Output file: {args.output}')

    data = []
    max_len = 0
    min_len = float('inf')
    sample_rate = None
    
    for f in args.files:
        if not os.path.isfile(f):
            print(f'Error. Can\'t find file: {f}. Check paths.')
            exit(1)
        
        print(f'Reading file: {f}')
        wav, sr = torchaudio.load(f)
        
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            print(f'Warning: Sample rate mismatch. Expected {sample_rate}, got {sr}. Resampling...')
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            wav = resampler(wav)

        # Ensure stereo (2 channels)
        if wav.shape[0] == 1:
            print("Mono detected. Converting to stereo by duplication.")
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            print(f"Multi-channel audio detected ({wav.shape[0]} channels). Using first 2 channels.")
            wav = wav[:2]

        # Move to device
        wav = wav.to(device)
        
        print(f"Waveform shape: {wav.shape} sample rate: {sr}")
        data.append(wav)
        max_len = max(max_len, wav.shape[1])
        min_len = min(min_len, wav.shape[1])

    # Choose target length based on argument
    target_len = min_len if args.trim_to_shortest else max_len
    target_tensor = torch.zeros(2, target_len, device=device)
    data = [match_tensor_shapes(wav, target_tensor) for wav in data]

    print(f"Target length: {target_len} ({'shortest' if args.trim_to_shortest else 'longest'} input file)")

    print("Starting ensemble processing...")
    result = average_waveforms(data, weights, args.type, args.n_fft, args.hop_length)
    
    print(f'Result shape: {result.shape}')
    
    # Save the result
    save_audio(result, sample_rate, args.output)
    print(f'Ensemble saved to: {args.output}')


if __name__ == "__main__":
    ensemble_files(None)
