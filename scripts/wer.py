#!/usr/bin/env python3
"""
Compute WER on the overlapping part between a truncated audio chunk
(audio_{i}.wav) and its full-length ground-truth utterance
(reconstructed_target_{i}.wav).

Data layout expected
--------------------
/mike_migrate2/codec-source-sep/test_results/
    ├── audio_0.wav
    ├── reconstructed_target_0.wav
    ├── ...
    └── text.txt          # 3 000 lines, one per utterance

Usage
-----
$ python evaluate_partial_wer.py \
      --test_dir /mike_migrate2/codec-source-sep/test_results \
      --device cuda      # or "cpu"

Dependencies
------------
pip install whisper-timestamped jiwer soundfile torch
"""

from pathlib import Path
import argparse

import torch
import soundfile as sf
import whisper_timestamped as whisper
from whisper_timestamped.transcribe import transcribe_timestamped
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip


def read_duration(wav_path: Path) -> float:
    """Return audio duration in seconds (uses soundfile)."""
    data, sr = sf.read(wav_path)
    return len(data) / sr


def join_words(segments):
    """Concatenate the 'text' field of all word-level dictionaries."""
    return " ".join(
        w["text"] for seg in segments for w in seg["words"]
    ).strip()


def overlap_reference(model, gt_wav: Path, chunk_seconds: float, text) -> str:
    """Run Whisper on the ground-truth audio and keep only words
    whose *end* timestamp is within `chunk_seconds`.
    """
    result = transcribe_timestamped(
        model,
        str(gt_wav),
        language="en",
        # timestamp_format="word",   # forces word-level timing
        # batch_size=1,
    )

    n_words_in_window = 0
    for seg in result["segments"]:
        for w in seg["words"]:
            if w["end"] <= chunk_seconds:   # 0.40-s margin
                n_words_in_window += 1
            else:
                break
        if seg["end"] > chunk_seconds :
            break

    # ---------- 4.  Slice the GT sentence ----------
    return " ".join(text.split()[:n_words_in_window]).strip()



def main(args):
    test_dir = Path(args.test_dir)
    gt_texts = (test_dir / "text.txt").read_text(encoding="utf-8").splitlines()

    print("Loading Whisper-timestamped ‘base’…")
    model = whisper.load_model("base", device=args.device)

    # WER uses case-insensitive, punctuation-free comparison
    normalise = Compose(
        [ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()]
    )

    sample_wers = []

    for i, text in enumerate(gt_texts[:200]):
        wav_chunk = test_dir / f"out_waveform_{i}.wav" # out_waveform_spkrbeam
        wav_gt = test_dir / f"reconstructed_target_{i}.wav"

        if not wav_chunk.is_file() or not wav_gt.is_file():
            print(f"[{i:04d}] Missing audio pair – skipped")
            continue

        # 1️⃣  Duration of the enhanced *chunk*
        chunk_len = read_duration(wav_chunk)

        # 2️⃣  Overlapping *reference* obtained from GT timestamps
        ref_overlap = overlap_reference(model, wav_gt, chunk_len, text)
        if not ref_overlap:
            print(f"[{i:04d}] No words inside the overlapping window – skipped")
            continue

        # 3️⃣  Transcribe the *chunk*
        hyp_result = transcribe_timestamped(
            model,
            str(wav_chunk),
            language="en",
            # timestamp_format="word",
            # batch_size=1,
        )
        hyp = join_words(hyp_result["segments"])

        # 4️⃣  Compute WER on overlapping part only
        print(normalise(ref_overlap), '----', normalise(hyp))
        wer_value = wer(normalise(ref_overlap), normalise(hyp))
        sample_wers.append(wer_value)

        print(
            f"[{i:04d}] WER={wer_value:.3f}  "
            f"({len(ref_overlap.split())} ref words, {len(hyp.split())} hyp words)"
        )

    # 🔚  Aggregate
    if sample_wers:
        mean_wer = sum(sample_wers) / len(sample_wers)
        print(
            f"\nProcessed {len(sample_wers)} files | "
            f"Mean WER on overlapping segment: {mean_wer:.2%}"
        )
    else:
        print("No valid samples processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir",
        default="/mike_migrate2/codec-source-sep/test_results",
        help="Folder containing audio_*.wav, reconstructed_target_*.wav and text.txt",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="'cuda' or 'cpu'",
    )
    main(parser.parse_args())
