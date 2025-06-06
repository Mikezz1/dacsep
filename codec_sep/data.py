import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset

USE_SPEAKER_EMBEDDER = False


class LibrimixDataset(Dataset):
    """
    Returns
        mix, target, reference, orig_len, speaker_idx, transcript_target, transcript_mix

        • mix              – mixture waveform               (1, cut_len)
        • target           – target-speaker waveform        (1, cut_len)
        • reference        – другой utterance того же спикера
        • orig_len         – длина target до crop/pad
        • speaker_idx      – числовой ID спикера-таргета
        • transcript_target – LibriSpeech-транскрипция target-аудио
        • transcript_mix   – транскрипция для микса
                             (конкатенация двух исходных реплик “spk1 spk2”)
    """

    LIBRISPEECH_SPLITS = ("test-clean",)

    def __init__(
        self,
        data_dir: str,
        manifest: str | Path | None = None,
        cut_len: int = 2 * 24_000,  # 2 s @ 24 kHz
        sample_rate_tgt: int = 24_000,
        return_second_speaker_no_mix: bool = False,
        librispeech_dir: str | Path = "/mike_migrate2/data_16khz/LibriSpeech",
    ):
        super().__init__()

        self.cut_len = cut_len
        self.sr_tgt = sample_rate_tgt
        self.root = Path(data_dir)
        self.mix_dir = self.root / "mix_clean"
        self.s1_dir = self.root / "s1"
        self.s2_dir = self.root / "s2"
        self.return_second_speaker_no_mix = return_second_speaker_no_mix
        self.spkr_id_to_num: dict[str, int] = {}

        self.librispeech_dir = Path(librispeech_dir).expanduser().resolve()

        manifest = Path(manifest or self.root / "manifest.csv")
        with open(manifest) as f:
            self.rows = list(csv.DictReader(f))

        self.spk2utts: dict[str, list[Path]] = defaultdict(list)

        spk_chap_pairs: set[tuple[str, str]] = set()
        idx_ = 0
        for row in self.rows:
            mix_id = row["mixture_ID"]
            part1, part2 = mix_id.split("_")

            spk1_id, chap1, _ = part1.split("-")
            spk2_id, chap2, _ = part2.split("-")

            # numerate speakers
            if spk1_id not in self.spkr_id_to_num:
                self.spkr_id_to_num[spk1_id] = idx_
                idx_ += 1
            if spk2_id not in self.spkr_id_to_num:
                self.spkr_id_to_num[spk2_id] = idx_
                idx_ += 1

            # index utterances
            self.spk2utts[spk1_id].append(self.s1_dir / f"{mix_id}.wav")
            self.spk2utts[spk2_id].append(self.s2_dir / f"{mix_id}.wav")

            # collect for transcript loading
            spk_chap_pairs.add((spk1_id, chap1))
            spk_chap_pairs.add((spk2_id, chap2))

        for lst in self.spk2utts.values():
            lst.sort()

        self._build_transcript_index(spk_chap_pairs)

    def _build_transcript_index(self, spk_chap_pairs: set[tuple[str, str]]) -> None:
        """
        Создаёт dict:  utterance_id (str «123-456-0007») -> transcript (str)
        Читаем только те speaker-chapter, что реально встречаются в манифесте.
        """
        self.transcripts: dict[str, str] = {}
        missing_pairs: list[tuple[str, str]] = []

        for spk_id, chap_id in sorted(spk_chap_pairs):
            rel = Path(f"{spk_id}/{chap_id}/{spk_id}-{chap_id}.trans.txt")

            found: Optional[Path] = None
            for split in self.LIBRISPEECH_SPLITS:
                cand = self.librispeech_dir / split / rel
                if cand.exists():
                    found = cand
                    break

            if found is None:
                missing_pairs.append((spk_id, chap_id))
                continue

            with open(found, "r", encoding="utf-8") as f:
                for line in f:
                    utt_id, text = line.rstrip().split(" ", 1)
                    self.transcripts[utt_id] = " ".join(text.split())

        if missing_pairs:
            print(
                f"[LibriMix-Dataset] Warning: no trans.txt for {len(missing_pairs)} "
                f"speaker-chapter pairs (examples: {missing_pairs[:3]})"
            )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # (1, n)
        return wav.squeeze(0), sr

    def _resample(self, wav: torch.Tensor, sr_from: int, sr_to: int) -> torch.Tensor:
        return (
            wav
            if sr_from == sr_to
            else torchaudio.functional.resample(wav, sr_from, sr_to)
        )

    def _uttid_from_part(self, part: str) -> str:
        """
        '4077-13754-0001' → '4077-13754-0001'
        (без изменения, но обособлено на случай дополнительных правил)
        """
        return part

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        mix_id = row["mixture_ID"]  #  '3536-8226-0026_1673-…'
        part1, part2 = mix_id.split("_")

        spk1_id = part1.split("-")[0]
        spk2_id = part2.split("-")[0]

        if (random.random() < 0.5) or (self.cut_len is None):
            target_path = self.s1_dir / f"{mix_id}.wav"
            spk_id = spk1_id
            target_part = part1
            mix_companion_part = part2
        else:
            target_path = self.s2_dir / f"{mix_id}.wav"
            spk_id = spk2_id
            target_part = part2
            mix_companion_part = part1

        if not self.return_second_speaker_no_mix:
            mix_path = self.mix_dir / f"{mix_id}.wav"
        else:
            mix_path = (
                self.s1_dir / f"{mix_id}.wav"
                if target_part == part2
                else self.s2_dir / f"{mix_id}.wav"
            )

        transcript_target = self.transcripts.get(self._uttid_from_part(target_part), "")
        transcript_mix = (
            self.transcripts.get(self._uttid_from_part(part1), "")
            + " "
            + self.transcripts.get(self._uttid_from_part(part2), "")
        ).strip()

        utts = self.spk2utts[spk_id]
        if len(utts) == 1:
            ref_path = utts[0]
        else:
            ref_path = random.choice([p for p in utts if p != target_path])

        target, sr_t = self._load(target_path)
        mix, sr_m = self._load(mix_path)
        reference, sr_r = self._load(ref_path)

        target = self._resample(target, sr_t, self.sr_tgt)
        mix = self._resample(mix, sr_m, self.sr_tgt)
        reference = self._resample(reference, sr_r, self.sr_tgt)

        length = target.size(0)
        if self.cut_len is not None:
            if length < self.cut_len:  # pad
                units = self.cut_len // length
                target = torch.cat(
                    [target] * units + [target[: self.cut_len % length]], -1
                )
                mix = torch.cat([mix] * units + [mix[: self.cut_len % length]], -1)
            else:  # random crop
                start = random.randint(0, length - self.cut_len)
                target = target[start : start + self.cut_len]
                mix = mix[start : start + self.cut_len]

            length_ref = reference.size(0)
            if length_ref < self.cut_len:
                units = self.cut_len // length_ref
                reference = torch.cat(
                    [reference] * units + [reference[: self.cut_len % length_ref]], -1
                )
            else:
                start = random.randint(0, length_ref - self.cut_len)
                reference = reference[start : start + self.cut_len]

        return (
            mix,
            target,
            reference,
            length,
            self.spkr_id_to_num[spk_id],
            transcript_target,
        )
