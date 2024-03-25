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

import base64
import os
from typing import Dict

from geniusrise import State, StreamingInput, StreamingOutput

from geniusrise_audio.s2t.inference import SpeechToTextInference
from geniusrise_audio.s2t.util import decode_audio


class SpeechToTextKafka(SpeechToTextInference):
    """
    SpeechToTextSpark leverages Apache Spark and PyTorch's distributed computing capabilities
    to perform speech-to-text inference on a large scale. It inherits from SpeechToTextInference
    and utilizes Spark's pandas UDFs for efficient distributed inference.
    """

    def __init__(
        self,
        input: StreamingInput,
        output: StreamingOutput,
        state: State,
        model_name: str,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = False,
        compile: bool = False,
        batch_size: int = 8,
        model_sampling_rate: int = 16_000,
        chunk_size: int = 0,
        overlap_size: int = 0,
        num_gpus: int = 1,
        **model_args,
    ):
        """
        Initialize the SpeechToTextSpark class with necessary configurations.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            spark_session (SparkSession): Active Spark session.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        self.model_name = model_name
        self.model_class = model_class
        self.processor_class = processor_class
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.compile = compile
        self.batch_size = batch_size
        self.model_sampling_rate = model_sampling_rate
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.model_args = model_args
        self.num_gpus = num_gpus

    def prepare(self):
        # Load models and processors as defined in the base class
        return self.load_models(
            model_name=self.model_name,
            processor_name=self.processor_name,
            model_revision=self.model_revision,
            processor_revision=self.processor_revision,
            model_class=self.model_class,
            processor_class=self.processor_class,
            use_cuda=self.use_cuda,
            precision=self.precision,
            quantization=self.quantization,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            compile=self.compile,
            use_whisper_cpp=self.use_whisper_cpp,
            use_faster_whisper=self.use_faster_whisper,
            **self.model_args,
        )

    def transcribe_stream(self):
        self.model, self.processor = self.prepare()

        for data in self.input.get():
            try:
                audio_data = data["audio"]  # Assuming row format aligns with df.select(audio_col).rdd
                metadata = data
                del metadata["audio"]
                audio_bytes = base64.b64decode(audio_data)
                audio_input, _ = decode_audio(audio_bytes=audio_bytes)

                # Process the audio input to get transcription
                transcription_result = self.process_audio(audio_input, model_sampling_rate=self.model_sampling_rate)
                transcription_result["metadata"] = metadata
            except Exception as e:
                print(e)
                transcription_result = {"transcription": "", "segments": [], "metadata": data}

            self.output.save(transcription_result)

    def process_audio(self, audio_input: bytes, model_sampling_rate: int) -> str:
        """
        Helper method to process audio data and return transcription.

        Args:
            audio_input (bytes): Audio data to be processed.
            model_sampling_rate (int): Sampling rate of the audio model.

        Returns:
            str: Transcribed text from audio data.
        """
        if self.use_whisper_cpp:
            transcription = self.model.transcribe(audio_input, num_proc=os.cpu_count())
        elif self.use_faster_whisper:
            transcription = self.process_faster_whisper(
                audio_input, model_sampling_rate, self.chunk_size, self.generation_args
            )
        else:
            # Default to using the whisper model processing
            transcription = self.process_whisper(
                audio_input,
                model_sampling_rate,
                self.processor_args,
                self.chunk_size,
                self.overlap_size,
                self.generation_args,
            )

        return transcription
