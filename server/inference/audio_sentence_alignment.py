# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import gc
import io
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List

import torch
import torchaudio
import torchaudio.functional as audio_F

from .align_utils import get_spans, load_model_dict, merge_repeats, time_to_frame
from .audio_reading_tools import wav_to_bytes

# Global logger for this module
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AudioAlignmentConfig:
    model_path_name: str = ""
    emission_interval: int = 30
    audio_format: str = "flac"
    use_star: bool = False
    device: str = "cuda"


class AudioAlignment:
    """Thread-safe singleton for audio-text alignment."""

    _instance = None
    _lock = threading.Lock()

    scale: int = 1000

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(AudioAlignment, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the singleton instance (called only once)."""
        logger.info("Initializing AudioAlignment model...")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = AudioAlignmentConfig(
            device=str(device),
            use_star=False,  # Set to False for standard alignment
        )

        self.config = config

        # FIXME: pass model name correctly
        logger.info("Loading forced alignment model and dictionary...")
        self.model, self.dictionary = load_model_dict()
        self.device = torch.device(config.device)
        self.model.to(self.device)

        if self.config.use_star:
            self.dictionary["<star>"] = len(self.dictionary)

        self.blank = self.dictionary["<blank>"]
        self.inverse_dictionary = {v: k for k, v in self.dictionary.items()}

        logger.info(
            f"AudioAlignment model loaded successfully on device: {self.device}"
        )

    @torch.inference_mode()
    def generate_emissions(self, waveform: torch.Tensor, reading_sr):
        emission_interval = self.config.emission_interval
        total_duration = waveform.size(1) / reading_sr

        emissions_arr = []

        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + emission_interval)

            context = emission_interval * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(reading_sr * input_start_time) : int(reading_sr * (input_end_time)),
            ]

            model_outs, _ = self.model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += emission_interval

        emissions = torch.cat(emissions_arr, dim=0).squeeze()
        emissions = torch.log_softmax(emissions, dim=-1)

        stride = float(waveform.size(1) * self.scale / emissions.size(0) / reading_sr)

        return emissions, stride

    @torch.inference_mode()
    def get_one_row_alignments(
        self, audio_arr, reading_sr, tokens: List[str]
    ) -> List[Dict]:
        """Internal method to perform forced alignment."""
        buffer = audio_arr.tobytes()
        waveform, audio_sf = torchaudio.load(io.BytesIO(buffer))
        waveform = waveform.to(self.device)
        assert audio_sf == reading_sr

        emissions, stride = self.generate_emissions(waveform, reading_sr)
        waveform = waveform.cpu()

        if self.config.use_star:
            T, _ = emissions.size()
            emissions = torch.cat(
                [emissions, torch.zeros(T, 1, device=self.device)], dim=1
            )

        if self.config.use_star:
            tokens = ["<star>"] + tokens

        token_indices = [
            self.dictionary[c]
            for c in " ".join(tokens).split(" ")
            if c in self.dictionary
        ]

        targets = torch.tensor(token_indices, dtype=torch.int32, device=self.device)

        input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
        target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)

        path, _ = audio_F.forced_align(
            emissions.unsqueeze(0),
            targets.unsqueeze(0),
            input_lengths,
            target_lengths,
            blank=self.blank,
        )
        path = path.squeeze().to("cpu").tolist()

        segments = merge_repeats(path, self.inverse_dictionary)

        spans = get_spans(tokens, segments)

        audio_segments = []
        for span in spans:
            seg_start_idx, seg_end_idx = span[0].start, span[-1].end
            segment_start_sec = seg_start_idx * stride / self.scale
            segment_end_sec = seg_end_idx * stride / self.scale
            start_frame = int(segment_start_sec * reading_sr)
            end_frame = int(segment_end_sec * reading_sr)
            trimmed_waveform = waveform[:, start_frame:end_frame]

            audio_segments.append(
                {
                    "segment_start_sec": segment_start_sec,
                    "segment_end_sec": segment_end_sec,
                    "segment_duration": segment_end_sec - segment_start_sec,
                    "segment_audio_bytes": wav_to_bytes(
                        trimmed_waveform, reading_sr, self.config.audio_format
                    ),
                }
            )
        return audio_segments
