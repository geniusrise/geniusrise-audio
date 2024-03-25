# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
from typing import Any, Dict, Optional, Tuple, Union

import torch
import transformers
from faster_whisper import WhisperModel
from geniusrise import Bolt, State, StreamingInput, StreamingOutput
from geniusrise.logging import setup_logger
from optimum.bettertransformer import BetterTransformer
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification
from whispercpp import Whisper


class AudioStream(Bolt):
    def __init__(
        self,
        input: StreamingInput,
        output: StreamingOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the AudioBulk with configurations and sets up logging.
        Prepares the environment for audio processing tasks.

        Args:
            input (BatchInput): The input data configuration for the audio processing task.
            output (BatchOutput): The output data configuration for the results of the audio processing.
            state (State): The state configuration for the Bolt, managing its operational status.
            **kwargs: Additional keyword arguments for extended functionality and model configurations.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def load_models(
        self,
        model_name: str,
        processor_name: str,
        model_location: Optional[str] = None,
        model_revision: Optional[str] = None,
        processor_revision: Optional[str] = None,
        model_class: str = "",
        processor_class: str = "AutoFeatureExtractor",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: Union[str, Dict, None] = "auto",
        max_memory: Dict[int, str] = {0: "24GB"},
        torchscript: bool = False,
        compile: bool = False,
        flash_attention: bool = False,
        better_transformers: bool = False,
        use_whisper_cpp: bool = False,
        use_faster_whisper: bool = False,
        **model_args: Any,
    ) -> Tuple[AutoModelForAudioClassification, AutoFeatureExtractor]:
        """
        Loads and configures the specified audio model and processor for audio processing.

        Args:
            model_name (str): Name or path of the audio model to load.
            processor_name (str): Name or path of the processor to load.
            model_revision (Optional[str]): Specific model revision to load (e.g., commit hash).
            processor_revision (Optional[str]): Specific processor revision to load.
            model_class (str): Class of the model to be loaded.
            processor_class (str): Class of the processor to be loaded.
            use_cuda (bool): Flag to use CUDA for GPU acceleration.
            precision (str): Desired precision for computations ("float32", "float16", etc.).
            quantization (int): Bit level for model quantization (0 for none, 8 for 8-bit).
            device_map (Union[str, Dict, None]): Specific device(s) for model operations.
            max_memory (Dict[int, str]): Maximum memory allocation for the model.
            torchscript (bool): Enable TorchScript for model optimization.
            compile (bool): Enable Torch JIT compilation.
            flash_attention (bool): Flag to enable Flash Attention optimization for faster processing.
            better_transformers (bool): Flag to enable Better Transformers optimization for faster processing.
            use_whisper_cpp (bool): Whether to use whisper.cpp to load the model. Defaults to False. Note: only works for these models: https://github.com/aarnphm/whispercpp/blob/524dd6f34e9d18137085fb92a42f1c31c9c6bc29/src/whispercpp/utils.py#L32
            use_faster_whisper (bool): Whether to use faster-whisper.
            **model_args (Any): Additional arguments for model loading.

        Returns:
            Tuple[AutoModelForAudioClassification, AutoFeatureExtractor]: Loaded model and processor.
        """
        self.log.info(f"Loading audio model: {model_name}")

        if use_whisper_cpp:
            return (
                self.load_models_whisper_cpp(
                    model_name=self.model_name,
                    basedir=self.output.output_folder,
                ),
                None,
            )
        elif use_faster_whisper:
            return (
                self.load_models_faster_whisper(
                    model_name=model_name,
                    device_map=device_map if type(device_map) is str else "auto",
                    precision=precision,
                    cpu_threads=multiprocessing.cpu_count(),
                    num_workers=1,
                    dowload_root=None,
                ),
                None,
            )

        # Determine torch dtype based on precision
        torch_dtype = self._get_torch_dtype(precision)

        # Configure device map for CUDA
        if use_cuda and not device_map:
            device_map = "auto"

        # Load the model and processor
        FeatureExtractorClass = getattr(transformers, processor_class)
        config = AutoConfig.from_pretrained(processor_name, revision=processor_revision)

        if model_name == "local":
            processor = FeatureExtractorClass.from_pretrained(model_location, torch_dtype=torch_dtype)
        else:
            processor = FeatureExtractorClass.from_pretrained(
                processor_name, revision=processor_revision, torch_dtype=torch_dtype
            )

        ModelClass = getattr(transformers, model_class)
        if quantization == 8:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    model_location,
                    max_memory=max_memory,
                    load_in_8bit=True,
                    config=config,
                    **model_args,
                )
            else:
                model = ModelClass.from_pretrained(
                    model_name,
                    revision=model_revision,
                    max_memory=max_memory,
                    load_in_8bit=True,
                    config=config,
                    **model_args,
                )
        elif quantization == 4:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    model_location,
                    max_memory=max_memory,
                    load_in_4bit=True,
                    config=config,
                    **model_args,
                )
            else:
                model = ModelClass.from_pretrained(
                    model_name,
                    revision=model_revision,
                    max_memory=max_memory,
                    load_in_4bit=True,
                    config=config,
                    **model_args,
                )
        else:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    model_location,
                    torch_dtype=torch_dtype,
                    max_memory=max_memory,
                    config=config,
                    **model_args,
                )
            else:
                model = ModelClass.from_pretrained(
                    model_name,
                    revision=model_revision,
                    torch_dtype=torch_dtype,
                    max_memory=max_memory,
                    config=config,
                    **model_args,
                )

        model = model.to(device_map)
        if compile:
            model = torch.compile(model)

        if better_transformers:
            model = BetterTransformer.transform(model, keep_original_model=True)

        # Set to evaluation mode for inference
        model.eval()

        self.log.debug("Audio model and processor loaded successfully.")
        return model, processor

    def load_models_whisper_cpp(self, model_name: str, basedir: str):
        return Whisper.from_pretrained(
            model_name=model_name,
            basedir=basedir,
        )

    def load_models_faster_whisper(
        self,
        model_name,
        device_map: str = "auto",
        precision="float16",
        quantization=0,
        cpu_threads=4,
        num_workers=1,
        dowload_root=None,
    ):
        return WhisperModel(
            model_size_or_path=model_name,
            device=device_map.split(":")[0] if ":" in device_map else device_map,
            device_index=int(device_map.replace("cuda:", "").replace("mps:", "")) if "cuda:" in device_map else 0,
            compute_type=precision if quantization == 0 else f"int{quantization}_{precision}",
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=dowload_root,
            local_files_only=False,
        )

    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """
        Determines the torch dtype based on the specified precision.

        Args:
            precision (str): The desired precision for computations.

        Returns:
            torch.dtype: The corresponding torch dtype.

        Raises:
            ValueError: If an unsupported precision is specified.
        """
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float,
            "float64": torch.float64,
            "double": torch.double,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "half": torch.half,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "short": torch.short,
            "int32": torch.int32,
            "int": torch.int,
            "int64": torch.int64,
            "quint8": torch.quint8,
            "qint8": torch.qint8,
            "qint32": torch.qint32,
        }
        return dtype_map.get(precision, torch.float)
