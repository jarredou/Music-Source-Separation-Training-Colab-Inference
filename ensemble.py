# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'  # forked by jarredou

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import librosa


def stft(wave, n_fft, hop_length):
    """
    Compute STFT for stereo audio using PyTorch
    wave: (channels, length)
    """
    return torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=wave.device),
        return_complex=True
    )


def istft(spec, hop_length, length):
    """
    Inverse STFT for stereo audio
    spec: (channels, freq, time)
    """
    n_fft = (spec.shape[1] - 1) * 2
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=spec.device),
        length=length
    )


def absmax(a, *, dim):
    abs_a = torch.abs(a)
    idx = torch.argmax(abs_a, dim=dim, keepdim=True)
    return torch.gather(a, dim, idx).squeeze(dim)


def lambda_max(arr, dim=None, key=None):
    key = key or (lambda x: x)
    idx = torch.argmax(key(arr), dim=dim, keepdim=True)
    return torch.gather(arr, dim, idx).squeeze(dim)


def lambda_min(arr, dim=None, key=None):
    key = key or (lambda x: x)
    idx = torch.argmin(key(arr), dim=dim, keepdim=True)
    return torch.gather(arr, dim, idx).squeeze(dim)


def match_tensor_shapes(tensor_1, tensor_2):
    target_length = tensor_2.shape[-1]
    current_length = tensor_1.shape[-1]

    if current_length > target_length:
        return tensor_1[..., :target_length]
    elif current_length < target_length:
        return F.pad(tensor_1, (0, target_length - current_length))
    return tensor_1


def average_waveforms(pred_track, weights, algorithm, n_fft, hop_length):
    pred_track = torch.stack(pred_track)
    final_length = pred_track.shape[-1]
    weights = torch.tensor(weights, device=pred_track.device, dtype=pred_track.dtype)

    if algorithm == 'avg_wave':
        weighted = pred_track * weights[:, None, None]
        return weighted.sum(0) / weights.sum()

    if algorithm == 'median_wave':
        return torch.median(pred_track, dim=0)[0]

    if algorithm == 'min_wave':
        return lambda_min(pred_track, dim=0, key=torch.abs)

    if algorithm == 'max_wave':
        return lambda_max(pred_track, dim=0, key=torch.abs)

    # FFT-domain methods
    specs = torch.stack([stft(x, n_fft, hop_length) for x in pred_track])

    if algorithm == 'avg_fft':
        weighted = specs * weights[:, None, None, None]
        avg_spec = weighted.sum(0) / weights.sum()
        return istft(avg_spec, hop_length, final_length)

    if algorithm == 'median_fft':
        return istft(torch.median(specs, dim=0)[0], hop_length, final_length)

    if algorithm == 'min_fft':
        return istft(lambda_min(specs, dim=0, key=torch.abs), hop_length, final_length)

    if algorithm == 'max_fft':
        return istft(absmax(specs, dim=0), hop_length, final_length)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def save_audio(waveform, sample_rate, output_path):
    output_path = Path(output_path)

    if waveform.is_cuda:
        waveform = waveform.cpu()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    audio_np = waveform.numpy().T  # (samples, channels)

    if output_path.suffix.lower() == '.flac':
        peak = abs(audio_np).max()
        if peak > 1.0:
            audio_np /= peak
        sf.write(output_path, audio_np, sample_rate, subtype='PCM_24')
    else:
        sf.write(output_path, audio_np, sample_rate, subtype='FLOAT')


def load_audio(path, target_sr=None):
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.T  # (channels, samples)

    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=1)
        sr = target_sr

    if audio.shape[0] == 1:
        audio = audio.repeat(2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    return torch.from_numpy(audio).float(), sr


def ensemble_files(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True)
    parser.add_argument("--type", default="avg_wave")
    parser.add_argument("--weights", nargs='+', type=float)
    parser.add_argument("--output", default="res.wav")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trim_to_shortest", action="store_true")

    args = parser.parse_args(args)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    weights = args.weights or [1.0] * len(args.files)

    data = []
    sample_rate = None
    min_len, max_len = float('inf'), 0

    for f in args.files:
        wav, sr = load_audio(f, sample_rate)
        sample_rate = sr if sample_rate is None else sample_rate

        wav = wav.to(device)
        data.append(wav)

        min_len = min(min_len, wav.shape[1])
        max_len = max(max_len, wav.shape[1])

    target_len = min_len if args.trim_to_shortest else max_len
    target = torch.zeros(2, target_len, device=device)

    data = [match_tensor_shapes(w, target) for w in data]

    result = average_waveforms(data, weights, args.type, args.n_fft, args.hop_length)
    save_audio(result, sample_rate, args.output)

    print(f"Saved ensemble to {args.output}")


if __name__ == "__main__":
    ensemble_files(None)
