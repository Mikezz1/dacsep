# import csv
# import random
# from collections import defaultdict
# from pathlib import Path

# import torch
# import torchaudio
# from torch.utils.data import Dataset

# USE_SPEAKER_EMBEDDER = False


# class LibrimixDataset(Dataset):
#     """
#     Returns (mix, target, reference, orig_len)
#         • mix        – mixture waveform        (1, cut_len)
#         • target     – target speaker waveform (1, cut_len)
#         • reference  – different utterance of the same speaker
#         • orig_len   – length of target before crop/pad
#     """

#     def __init__(
#         self,
#         data_dir: str,
#         manifest: str | Path | None = None,
#         cut_len: int = 2 * 24_000,  # 2 s @24 kHz
#         sample_rate_tgt: int = 24_000,  # network SR (mix / target / ref)
#         return_second_speaker_no_mix: bool = False,
#     ):
#         super().__init__()

#         self.cut_len = cut_len
#         self.sr_tgt = sample_rate_tgt
#         self.root = Path(data_dir)
#         self.mix_dir = self.root / "mix_clean"
#         self.s1_dir = self.root / "s1"
#         self.s2_dir = self.root / "s2"
#         self.return_second_speaker_no_mix = return_second_speaker_no_mix
#         self.spkr_id_to_num = {}

#         # ---------- 1) read manifest -------------------------------------
#         manifest = Path(manifest or self.root / "manifest.csv")
#         with open(manifest) as f:
#             self.rows = list(csv.DictReader(f))

#         # ---------- 2) speaker‑to‑utterance index (covers *both* speakers) ------
#         self.spk2utts: dict[str, list[Path]] = defaultdict(list)

#         idx_ = 0

#         for row in self.rows:  # each row == one mixture
#             mix_id = row["mixture_ID"]
#             part1, part2 = mix_id.split("_")

#             spk1_id = part1.split("-")[0]
#             spk2_id = part2.split("-")[0]
#             if spk1_id not in self.spkr_id_to_num:
#                 self.spkr_id_to_num[spk1_id] = idx_
#                 idx_ += 1

#             if spk2_id not in self.spkr_id_to_num:
#                 self.spkr_id_to_num[spk2_id] = idx_
#                 idx_ += 1

#             self.spk2utts[spk1_id].append(self.s1_dir / f"{mix_id}.wav")
#             self.spk2utts[spk2_id].append(self.s2_dir / f"{mix_id}.wav")

#         for lst in self.spk2utts.values():  # reproducibility
#             lst.sort()

#     # ---------------------------------------------------------------------
#     def __len__(self) -> int:
#         return len(self.rows)

#     # ---------------------------------------------------------------------
#     @staticmethod
#     def _load(path: Path) -> torch.Tensor:
#         wav, sr = torchaudio.load(path)  # (1, n), sr varies (8 k / 16 k)
#         return wav.squeeze(0), sr

#     # ---------------------------------------------------------------------
#     def _resample(self, wav: torch.Tensor, sr_from: int, sr_to: int) -> torch.Tensor:
#         return (
#             wav
#             if sr_from == sr_to
#             else torchaudio.functional.resample(wav, sr_from, sr_to)
#         )

#     def __getitem__(self, idx: int):
#         row = self.rows[idx]
#         mix_id = row["mixture_ID"]  # e.g. 3536‑8226‑0026_1673‑...
#         part1, part2 = mix_id.split("_")
#         spk1_id = part1.split("-")[0]
#         spk2_id = part2.split("-")[0]

#         if random.random() < 0.5:
#             target_path = self.s1_dir / f"{mix_id}.wav"
#             spk_id = spk1_id

#             mix_path = self.s2_dir / f"{mix_id}.wav"

#         else:
#             target_path = self.s2_dir / f"{mix_id}.wav"
#             spk_id = spk2_id

#             mix_path = self.s1_dir / f"{mix_id}.wav"

#         if not self.return_second_speaker_no_mix:

#             mix_path = self.mix_dir / f"{mix_id}.wav"

#         utts = self.spk2utts[spk_id]

#         if len(utts) == 1:  # defensive – rare with LibriMix
#             ref_path = utts[0]
#         else:
#             ref_path = random.choice([p for p in utts if p != target_path])

#         target, sr_t = self._load(target_path)
#         mix, sr_m = self._load(mix_path)
#         ref, sr_r = self._load(ref_path)

#         target = self._resample(target, sr_t, self.sr_tgt)
#         mix = self._resample(mix, sr_m, self.sr_tgt)
#         reference = self._resample(ref, sr_r, self.sr_tgt)

#         length = target.size(0)
#         if length < self.cut_len:
#             units = self.cut_len // length
#             target_ds_final = []
#             mix_ds_final = []
#             for i in range(units):
#                 target_ds_final.append(target)
#                 mix_ds_final.append(mix)
#             target_ds_final.append(target[: self.cut_len % length])
#             mix_ds_final.append(mix[: self.cut_len % length])
#             target = torch.cat(target_ds_final, dim=-1)
#             mix = torch.cat(mix_ds_final, dim=-1)
#         else:
#             # randomly cut 2 seconds segment
#             wav_start = random.randint(0, length - self.cut_len)
#             mix = mix[wav_start : wav_start + self.cut_len]
#             target = target[wav_start : wav_start + self.cut_len]

#         length_ref = reference.size(0)
#         if length_ref < self.cut_len:
#             units = self.cut_len // length_ref
#             reference_ds_final = []
#             for i in range(units):
#                 reference_ds_final.append(reference)
#             reference_ds_final.append(reference[: self.cut_len % length_ref])
#             reference = torch.cat(reference_ds_final, dim=-1)
#         else:
#             wav_start = random.randint(0, length_ref - self.cut_len)
#             reference = reference[wav_start : wav_start + self.cut_len]

#         return (
#             mix,
#             target,
#             reference,
#             length,
#             self.spkr_id_to_num[spk_id]
#         )



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

    LIBRISPEECH_SPLITS = (
        "test-clean",
    )

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

        # ---------- 1) read manifest -------------------------------------
        manifest = Path(manifest or self.root / "manifest.csv")
        with open(manifest) as f:
            self.rows = list(csv.DictReader(f))

        # ---------- 2) speaker-to-utterance index (covers *both* speakers) ------
        self.spk2utts: dict[str, list[Path]] = defaultdict(list)

        # собрать пары <speaker, chapter> для дальнейшего чтения trans.txt —
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

        for lst in self.spk2utts.values():  # reproducibility
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

        if (random.random() < 0.5) or  (self.cut_len is None):
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

        transcript_target = self.transcripts.get(
            self._uttid_from_part(target_part), ""
        )
        transcript_mix = (
            self.transcripts.get(self._uttid_from_part(part1), "")
            + " "
            + self.transcripts.get(self._uttid_from_part(part2), "")
        ).strip()

        utts = self.spk2utts[spk_id]
        if len(utts) == 1:  # defensive – редко бывает
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
                target = torch.cat([target] * units + [target[: self.cut_len % length]], -1)
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


# import os
# import glob
# import torch
# import torchaudio
# from natsort import natsorted
# from torch.utils.data import Dataset, DataLoader

# import torch.utils.data
# import torchaudio
# import os
# import random
# from itertools import groupby

# USE_SPEAKER_EMBEDDER = False


# class LibrimixDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir='/mike_migrate2/Libri2Mix/wav8k/min/train-100/', cut_len=24000 * 2):
#         self.cut_len = cut_len
#         self.mix_dir = os.path.join(data_dir, "mix_clean")
#         self.target_dir = os.path.join(data_dir, "s2")
#         self.target_wav_name = os.listdir(self.target_dir)
#         self.target_wav_name = natsorted(self.target_wav_name)
#         # speaker_mapping = [[v.split("-")[0], v] for v in self.target_wav_name]

#         # self.speaker_mapping = {}
#         # for sublist in speaker_mapping:
#         #     if sublist[0] in self.speaker_mapping:
#         #         self.speaker_mapping[sublist[0]].append(sublist[1])
#         #     else:
#         #         self.speaker_mapping[sublist[0]] = [sublist[1]]

#     def __len__(self):
#         return len(self.target_wav_name)

#     def __getitem__(self, idx):
#         target_file = os.path.join(self.target_dir, self.target_wav_name[idx])

#         target_speaker = self.target_wav_name[idx].split("-")[0]
#         # reference_files = self.speaker_mapping[target_speaker]
#         # reference_file = reference_files[random.randint(0, len(reference_files) - 1)]
#         # reference_file = os.path.join(self.target_dir, reference_file)

#         mix_file = os.path.join(self.mix_dir, self.target_wav_name[idx])

#         target_ds, _ = torchaudio.load(target_file)
#         mix_ds, _ = torchaudio.load(mix_file)
#         # reference_ds, _ = torchaudio.load(reference_file)
#         reference_ds = torchaudio.functional.resample(target_ds, 16000, 24000)
#         target_ds = torchaudio.functional.resample(target_ds, 16000, 24000)
#         mix_ds = torchaudio.functional.resample(mix_ds, 16000, 24000)

#         target_ds = target_ds.squeeze()
#         mix_ds = mix_ds.squeeze()
#         reference_ds = reference_ds.squeeze()
#         length = len(target_ds)
#         assert length == len(mix_ds)
#         if length < self.cut_len:
#             units = self.cut_len // length
#             target_ds_final = []
#             mix_ds_final = []
#             reference_ds_final = []
#             for i in range(units):
#                 target_ds_final.append(target_ds)
#                 mix_ds_final.append(mix_ds)
#                 reference_ds_final.append(reference_ds)
#             target_ds_final.append(target_ds[: self.cut_len % length])
#             mix_ds_final.append(mix_ds[: self.cut_len % length])
#             reference_ds_final.append(reference_ds[: self.cut_len % length])
#             target_ds = torch.cat(target_ds_final, dim=-1)
#             mix_ds = torch.cat(mix_ds_final, dim=-1)
#             reference_ds = torch.cat(reference_ds_final, dim=-1)
#         else:
#             # randomly cut 2 seconds segment
#             wav_start = random.randint(0, length - self.cut_len)
#             mix_ds = mix_ds[wav_start : wav_start + self.cut_len]
#             target_ds = target_ds[wav_start : wav_start + self.cut_len]

#             wav_start = random.randint(0, reference_ds.size(0) - self.cut_len )
#             reference_ds = reference_ds[wav_start : wav_start + self.cut_len ]

#         return mix_ds, target_ds, reference_ds, length


# # import csv
# # import random
# # from collections import defaultdict
# # from pathlib import Path

# # import torch
# # import torchaudio
# # from torch.utils.data import Dataset

# # USE_SPEAKER_EMBEDDER = False


# # class LibrimixDataset(Dataset):
# #     """
# #     Returns (mix, target, reference, orig_len)
# #         • mix        – mixture waveform        (1, cut_len)
# #         • target     – target speaker waveform (1, cut_len)
# #         • reference  – different utterance of the same speaker
# #         • orig_len   – length of target before crop/pad
# #     """

# #     def __init__(
# #         self,
# #         root_dir: str,
# #         manifest: str | Path | None = None,
# #         cut_len: int = 2 * 24_000,      # 2 s @24 kHz
# #         sample_rate_tgt: int = 24_000,  # network SR (mix / target / ref)
# #     ):
# #         super().__init__()

# #         self.cut_len  = cut_len
# #         self.sr_tgt   = sample_rate_tgt
# #         self.root     = Path(root_dir)
# #         self.mix_dir  = self.root / "mix_clean"
# #         self.s1_dir   = self.root / "s1"
# #         self.s2_dir   = self.root / "s2"

# #         # ---------- 1) read manifest -------------------------------------
# #         manifest = Path(manifest or self.root / "manifest.csv")
# #         with open(manifest) as f:
# #             self.rows = list(csv.DictReader(f))

# #         # ---------- 2) speaker‑to‑utterance index (covers *both* speakers) ------
# #         self.spk2utts: dict[str, list[Path]] = defaultdict(list)

# #         for row in self.rows:                       # each row == one mixture
# #             mix_id = row["mixture_ID"]
# #             part1, part2 = mix_id.split("_")

# #             spk1_id = part1.split("-")[0]
# #             spk2_id = part2.split("-")[0]

# #             self.spk2utts[spk1_id].append(self.s1_dir / f"{mix_id}.wav")
# #             self.spk2utts[spk2_id].append(self.s2_dir / f"{mix_id}.wav")

# #         for lst in self.spk2utts.values():          # reproducibility
# #             lst.sort()
# #     # ---------------------------------------------------------------------
# #     def __len__(self) -> int:
# #         return len(self.rows)

# #     # ---------------------------------------------------------------------
# #     @staticmethod
# #     def _load(path: Path) -> torch.Tensor:
# #         wav, sr = torchaudio.load(path)        # (1, n), sr varies (8 k / 16 k)
# #         return wav.squeeze(0), sr

# #     # ---------------------------------------------------------------------
# #     def _resample(self, wav: torch.Tensor, sr_from: int, sr_to: int) -> torch.Tensor:
# #         return (
# #             wav if sr_from == sr_to
# #             else torchaudio.functional.resample(wav, sr_from, sr_to)
# #         )

# #     # ---------------------------------------------------------------------
# #     def _crop_or_pad(self, sig: torch.Tensor) -> torch.Tensor:
# #         """Random crop; if too short, pad by repetition."""
# #         if sig.numel() >= self.cut_len:
# #             start = random.randint(0, sig.numel() - self.cut_len)
# #             return sig[start : start + self.cut_len]
# #         reps, extra = divmod(self.cut_len, sig.numel())
# #         out = sig.repeat(reps)
# #         return torch.cat([out, sig[:extra]]) if extra else out

# #     # ---------------------------------------------------------------------
# #     def __getitem__(self, idx: int):
# #         row       = self.rows[idx]
# #         mix_id    : str = row["mixture_ID"]                     # e.g. 3536‑8226‑0026_1673‑...
# #         part1, part2   = mix_id.split("_")
# #         spk1_id        = part1.split("-")[0]
# #         spk2_id        = part2.split("-")[0]

# #         # -------- randomly choose speaker 1 or 2 as the CURRENT target -----
# #         if random.random() < 0.5:
# #             target_path = self.s1_dir / f"{mix_id}.wav"
# #             spk_id      = spk1_id
# #         else:
# #             target_path = self.s2_dir / f"{mix_id}.wav"
# #             spk_id      = spk2_id

# #         mix_path = self.mix_dir / f"{mix_id}.wav"

# #         # -------- pick a *different* utterance of the same speaker ----------
# #         utts = self.spk2utts[spk_id]

# #         if len(utts) == 1:                   # defensive – rare with LibriMix
# #             ref_path = utts[0]
# #         else:

# #             ref_path = random.choice([p for p in utts if p != target_path])

# #         # -------- load + resample ------------------------------------------
# #         target, sr_t = self._load(target_path)
# #         mix,    sr_m = self._load(mix_path)
# #         ref,    sr_r = self._load(ref_path)

# #         target = self._resample(target, sr_t, self.sr_tgt)
# #         mix    = self._resample(mix,    sr_m, self.sr_tgt)
# #         ref_sr = 16_000 if USE_SPEAKER_EMBEDDER else self.sr_tgt
# #         reference = self._resample(ref, sr_r, ref_sr)

# #         # -------- crop / pad to fixed length --------------------------------
# #         target_len = target.numel()                       # before crop/pad
# #         target     = self._crop_or_pad(target)
# #         mix        = self._crop_or_pad(mix)
# #         reference  = self._crop_or_pad(reference)

# #         # final shapes: (cut_len,) → add channel dim here if you prefer (1, ⋅)
# #         return (
# #             mix,
# #             target,
# #             reference,
# #             target_len,
# #         )
