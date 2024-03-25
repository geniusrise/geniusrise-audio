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
from typing import Any, Dict

import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from geniusrise_audio.t2s.inference import TextToSpeechInference
from geniusrise_audio.t2s.util import convert_waveform_to_audio_file


class TextToSpeechAPI(TextToSpeechInference):
    r"""
    TextToSpeechAPI for converting text to speech using various TTS models.

    Attributes:
        model (AutoModelForSeq2SeqLM): The text-to-speech model.
        tokenizer (AutoTokenizer): The tokenizer for the model.

    Methods:
        synthesize(text_input: str) -> bytes:
            Converts the given text input to speech using the text-to-speech model.

    Example CLI Usage:

    ```
    genius TextToSpeechAPI rise \
    batch \
        --input_folder ./input \
    batch \
        --output_folder ./output \
    none \
        --id facebook/mms-tts-eng \
        listen \
            --args \
                model_name="facebook/mms-tts-eng" \
                model_class="VitsModel" \
                processor_class="VitsTokenizer" \
                use_cuda=True \
                precision="float32" \
                quantization=0 \
                device_map="cuda:0" \
                max_memory=None \
                torchscript=False \
                compile=False \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```
    """

    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the TextToSpeechAPI with configurations for text-to-speech processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)
        self.hf_pipeline = None
        self.vocoder = None
        self.embeddings_dataset = None

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def synthesize(self):
        """
        API endpoint to convert text input to speech using the text-to-speech model.
        Expects a JSON input with 'text' as a key containing the text to be synthesized.

        Returns:
            Dict[str, str]: A dictionary containing the base64 encoded audio data.

        Example CURL Request for synthesis:
        ... [Provide example CURL request] ...
        """
        input_json = cherrypy.request.json
        text_data = input_json.get("text")
        output_type = input_json.get("output_type")
        voice_preset = input_json.get("voice_preset")

        generate_args = input_json.copy()

        if "text" in generate_args:
            del generate_args["text"]
        if "output_type" in generate_args:
            del generate_args["output_type"]
        if "voice_preset" in generate_args:
            del generate_args["voice_preset"]

        if not text_data:
            raise cherrypy.HTTPError(400, "No text data provided.")

        # Perform inference
        if self.model.config.model_type == "vits":
            audio_output = self.process_mms(text_data, generate_args=generate_args)
        elif self.model.config.model_type == "coarse_acoustics" or self.model.config.model_type == "bark":
            audio_output = self.process_bark(text_data, voice_preset=voice_preset, generate_args=generate_args)
        elif self.model.config.model_type == "speecht5":
            audio_output = self.process_speecht5_tts(text_data, voice_preset=voice_preset, generate_args=generate_args)
        elif self.model.config.model_type == "seamless_m4t_v2":
            audio_output = self.process_seamless(text_data, voice_preset=voice_preset, generate_args=generate_args)

        # Convert audio to base64 encoded data
        sample_rate = (
            self.model.generation_config.sample_rate if hasattr(self.model.generation_config, "sample_rate") else 16_000
        )
        audio_file = convert_waveform_to_audio_file(audio_output, format=output_type, sample_rate=sample_rate)
        audio_base64 = base64.b64encode(audio_file)

        return {"audio_file": audio_base64.decode("utf-8"), "input": text_data}

    def initialize_pipeline(self):
        """
        Lazy initialization of the TTS Hugging Face pipeline.
        """
        if not self.hf_pipeline:
            self.hf_pipeline = pipeline(
                "text-to-speech",
                model=self.model,
                tokenizer=self.processor,
                framework="pt",
            )

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def tts_pipeline(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Converts text to speech using the Hugging Face pipeline.

        Args:
            **kwargs (Any): Arbitrary keyword arguments, typically containing 'text' for the input text.

        Returns:
            Dict[str, Any]: A dictionary containing the base64 encoded audio data.

        Example CURL Request for synthesis:
        ... [Provide example CURL request] ...
        """
        self.initialize_pipeline()  # Initialize the pipeline on first API hit

        input_json = cherrypy.request.json
        text_data = input_json.get("text")

        result = self.hf_pipeline(text_data, **kwargs)  # type: ignore

        # Convert audio to base64 encoded data
        audio_base64 = base64.encode(result)  # type: ignore

        return {"audio_file": audio_base64, "input": text_data}
