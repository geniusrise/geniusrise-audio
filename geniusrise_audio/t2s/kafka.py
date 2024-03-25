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
from typing import Dict

import numpy as np
from geniusrise import State, StreamingInput, StreamingOutput

from geniusrise_audio.t2s.inference import TextToSpeechInference


class TextToSpeechKafka(TextToSpeechInference):
    """
    TextToSpeechKafka leverages Apache Kafka for real-time text-to-speech inference.
    It inherits from TextToSpeechInference and processes text data from a Kafka input stream,
    generates speech synthesis, and sends the results to a Kafka output stream.
    """

    def __init__(
        self,
        input: StreamingInput,
        output: StreamingOutput,
        state: State,
        model_name: str,
        model_class: str = "AutoModelForSeq2SeqLM",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        device_map: str | Dict | None = "auto",
        **model_args,
    ):
        """
        Initialize the TextToSpeechKafka class with necessary configurations.

        Args:
            input (StreamingInput): The input data configuration.
            output (StreamingOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        self.model_name = model_name
        self.model_class = model_class
        self.processor_class = processor_class
        self.use_cuda = use_cuda
        self.device_map = device_map
        self.model_args = model_args

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
            device_map=self.device_map,
            **self.model_args,
        )

    def synthesize_stream(self):
        self.model, self.processor = self.prepare()

        for data in self.input.get():
            try:
                text_data = data["text"]
                voice_preset = data["voice_preset"]
                metadata = data
                del metadata["text"]
                del metadata["voice_preset"]

                # Process the text input to get speech synthesis
                synthesis_result = self.process_text(text_data, voice_preset)
                synthesis_result_base64 = base64.b64encode(synthesis_result.tobytes()).decode("utf-8")
                output_data = {"audio": synthesis_result_base64, "metadata": metadata}
            except Exception as e:
                print(e)
                output_data = {"audio": "", "metadata": data}

            self.output.save(output_data)

    def process_text(self, text_input: str, voice_preset: str) -> np.ndarray:
        """
        Helper method to process text data and return speech synthesis.

        Args:
            text_input (str): Text data to be processed.
            voice_preset (str): Voice preset to use for synthesis.

        Returns:
            np.ndarray: Synthesized speech audio data.
        """
        if "mms" in self.model_name.lower():
            audio_output = self.process_mms(text_input, self.generation_args)
        elif "bark" in self.model_name.lower():
            audio_output = self.process_bark(text_input, voice_preset, self.generation_args)
        elif "speecht5" in self.model_name.lower():
            audio_output = self.process_speecht5_tts(text_input, voice_preset, self.generation_args)
        else:
            # Default to using the seamless model processing
            audio_output = self.process_seamless(text_input, voice_preset, self.generation_args)

        return audio_output
