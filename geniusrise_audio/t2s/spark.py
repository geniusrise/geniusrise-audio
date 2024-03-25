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
from geniusrise import BatchInput, BatchOutput, State
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from geniusrise_audio.t2s.inference import TextToSpeechInference


class TextToSpeechSpark(TextToSpeechInference):
    """
    TextToSpeechSpark leverages Apache Spark and PyTorch's distributed computing capabilities
    to perform text-to-speech inference on a large scale. It inherits from TextToSpeechInference
    and utilizes Spark's pandas UDFs for efficient distributed inference.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        spark_session: SparkSession,
        model_name: str,
        model_class: str = "AutoModelForSeq2SeqLM",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        device_map: str | Dict | None = "auto",
        num_gpus: int = 1,
        **model_args,
    ):
        """
        Initialize the TextToSpeechSpark class with necessary configurations.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            spark_session (SparkSession): Active Spark session.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        self.spark = spark_session
        self.model_name = model_name
        self.model_class = model_class
        self.processor_class = processor_class
        self.use_cuda = use_cuda
        self.device_map = device_map
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
            device_map=self.device_map,
            **self.model_args,
        )

    def synthesize_dataframe(
        self,
        df: DataFrame,
        text_col: str,
        voice_preset_col: str,
    ) -> DataFrame:
        """
        Synthesize text data in a Spark DataFrame to speech using distributed processing.
        """
        # Add a unique ID to each row to use for joining later
        df_with_id = df.withColumn("row_id", monotonically_increasing_id())

        # Prepare and run the distributed inference
        distributor = TorchDistributor(local_mode=False, use_gpu=self.use_cuda, num_processes=self.num_gpus)

        def distributed_synthesize(iterator):
            import torch
            import torch.distributed

            torch.distributed.init_process_group(backend="nccl")

            # Load model and processor inside the distributed function
            self.model, self.processor = self.prepare()
            synthesis_results = []

            for row in iterator:
                try:
                    text_data = row[0]  # Assuming row format aligns with df.select(text_col).rdd
                    voice_preset = row[1]  # Assuming row format aligns with df.select(voice_preset_col).rdd

                    # Process the text input to get speech synthesis
                    synthesis_result = self.process_text(text_data, voice_preset)
                except Exception as e:
                    self.log.exception(e)
                    synthesis_result = np.zeros(1)

                synthesis_result = base64.b64encode(synthesis_result.tobytes()).decode("utf-8")
                synthesis_results.append(synthesis_result)

            torch.destroy_process_group()
            # convert list of numpy arrays to rdd
            return synthesis_results

        # Select only the IDs, text data, and voice preset, and apply the mapPartitions operation
        synthesized_rdd = distributor.run(
            distributed_synthesize, df_with_id.select("row_id", text_col, voice_preset_col).rdd
        )

        # Convert RDD back to DataFrame and rename columns appropriately
        synthesized_df = synthesized_rdd.toDF(["row_id", "audio"])

        # Join the synthesized DataFrame with the original DataFrame using the row_id
        result_df = df_with_id.join(synthesized_df, on="row_id").drop("row_id")

        # Convert audio numpy array to base64 string for storage
        result_df = result_df.withColumn("audio", result_df["audio"].cast("binary")).withColumn(
            "audio", result_df["audio"].cast("string")
        )

        return result_df

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
