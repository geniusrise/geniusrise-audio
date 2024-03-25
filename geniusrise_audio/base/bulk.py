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
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from faster_whisper import WhisperModel
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from optimum.bettertransformer import BetterTransformer
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoProcessor,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from whispercpp import Whisper

from geniusrise_audio.base.communication import send_email


class AudioBulk(Bolt):
    """
    AudioBulk is a class designed for bulk processing of audio data using various audio models from Hugging Face.
    It focuses on audio generation and transformation tasks, supporting a range of models and configurations.

    Attributes:
        model (AutoModelForAudioClassification): The audio model for generation or transformation tasks.
        processor (AutoFeatureExtractor): The processor for preparing input data for the model.

    Args:
        input (BatchInput): Configuration and data inputs for the batch process.
        output (BatchOutput): Configurations for output data handling.
        state (State): State management for the Bolt.
        **kwargs: Arbitrary keyword arguments for extended configurations.

    Methods:
        audio(**kwargs: Any) -> Dict[str, Any]:
            Provides an API endpoint for audio processing functionality.
            Accepts various parameters for customizing the audio processing tasks.

        process(audio_input: Union[str, bytes], **processing_params: Any) -> dict:
            Processes the audio input based on the provided parameters. Supports multiple processing methods.
    """

    model: AutoModelForAudioClassification
    processor: AutoFeatureExtractor | AutoProcessor

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
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

    def generate(
        self,
        prompt: str,
        decoding_strategy: str = "generate",
        **generation_params: Any,
    ) -> str:
        r"""
        Generate text completion for the given prompt using the specified decoding strategy.

        Args:
            prompt (str): The prompt to generate text completion for.
            decoding_strategy (str, optional): The decoding strategy to use. Defaults to "generate".
            **generation_params (Any): Additional parameters to pass to the decoding strategy.

        Returns:
            str: The generated text completion.

        Raises:
            Exception: If an error occurs during generation.

        Supported decoding strategies and their additional parameters:
            - "generate": Uses the model's default generation method. (Parameters: max_length, num_beams, etc.)
            - "greedy_search": Generates text using a greedy search decoding strategy.
            Parameters: max_length, eos_token_id, pad_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "contrastive_search": Generates text using contrastive search decoding strategy.
            Parameters: top_k, penalty_alpha, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus, sequential.
            - "sample": Generates text using a sampling decoding strategy.
            Parameters: do_sample, temperature, top_k, top_p, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "beam_search": Generates text using beam search decoding strategy.
            Parameters: num_beams, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "beam_sample": Generates text using beam search with sampling decoding strategy.
            Parameters: num_beams, temperature, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "group_beam_search": Generates text using group beam search decoding strategy.
            Parameters: num_beams, diversity_penalty, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "constrained_beam_search": Generates text using constrained beam search decoding strategy.
            Parameters: num_beams, max_length, constraints, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.

        All generation parameters:
            - max_length: Maximum length the generated tokens can have
            - max_new_tokens: Maximum number of tokens to generate, ignoring prompt tokens
            - min_length: Minimum length of the sequence to be generated
            - min_new_tokens: Minimum number of tokens to generate, ignoring prompt tokens
            - early_stopping: Stopping condition for beam-based methods
            - max_time: Maximum time allowed for computation in seconds
            - do_sample: Whether to use sampling for generation
            - num_beams: Number of beams for beam search
            - num_beam_groups: Number of groups for beam search to ensure diversity
            - penalty_alpha: Balances model confidence and degeneration penalty in contrastive search
            - use_cache: Whether the model should use past key/values attentions to speed up decoding
            - temperature: Modulates next token probabilities
            - top_k: Number of highest probability tokens to keep for top-k-filtering
            - top_p: Smallest set of most probable tokens with cumulative probability >= top_p
            - typical_p: Conditional probability of predicting a target token next
            - epsilon_cutoff: Tokens with a conditional probability > epsilon_cutoff will be sampled
            - eta_cutoff: Eta sampling, a hybrid of locally typical sampling and epsilon sampling
            - diversity_penalty: Penalty subtracted from a beam's score if it generates a token same as any other group
            - repetition_penalty: Penalty for repetition of ngrams
            - encoder_repetition_penalty: Penalty on sequences not in the original input
            - length_penalty: Exponential penalty to the length for beam-based generation
            - no_repeat_ngram_size: All ngrams of this size can only occur once
            - bad_words_ids: List of token ids that are not allowed to be generated
            - force_words_ids: List of token ids that must be generated
            - renormalize_logits: Renormalize the logits after applying all logits processors
            - constraints: Custom constraints for generation
            - forced_bos_token_id: Token ID to force as the first generated token
            - forced_eos_token_id: Token ID to force as the last generated token
            - remove_invalid_values: Remove possible NaN and inf outputs
            - exponential_decay_length_penalty: Exponentially increasing length penalty after a certain number of tokens
            - suppress_tokens: Tokens that will be suppressed during generation
            - begin_suppress_tokens: Tokens that will be suppressed at the beginning of generation
            - forced_decoder_ids: Mapping from generation indices to token indices that will be forced
            - sequence_bias: Maps a sequence of tokens to its bias term
            - guidance_scale: Guidance scale for classifier free guidance (CFG)
            - low_memory: Switch to sequential topk for contrastive search to reduce peak memory
            - num_return_sequences: Number of independently computed returned sequences for each batch element
            - output_attentions: Whether to return the attentions tensors of all layers
            - output_hidden_states: Whether to return the hidden states of all layers
            - output_scores: Whether to return the prediction scores
            - return_dict_in_generate: Whether to return a ModelOutput instead of a plain tuple
            - pad_token_id: The id of the padding token
            - bos_token_id: The id of the beginning-of-sequence token
            - eos_token_id: The id of the end-of-sequence token
            - max_length: The maximum length of the sequence to be generated
            - eos_token_id: End-of-sequence token ID
            - pad_token_id: Padding token ID
            - output_attentions: Return attention tensors of all attention layers if True
            - output_hidden_states: Return hidden states of all layers if True
            - output_scores: Return prediction scores if True
            - return_dict_in_generate: Return a ModelOutput instead of a plain tuple if True
            - synced_gpus: Continue running the while loop until max_length for ZeRO stage 3 if True
            - top_k: Size of the candidate set for re-ranking in contrastive search
            - penalty_alpha: Degeneration penalty; active when larger than 0
            - eos_token_id: End-of-sequence token ID(s)
            - sequential: Switch to sequential topk hidden state computation to reduce memory if True
            - do_sample: Use sampling for generation if True
            - temperature: Temperature for sampling
            - top_p: Cumulative probability for top-p-filtering
            - diversity_penalty: Penalty for reducing similarity across different beam groups
            - constraints: List of constraints to apply during beam search
            - synced_gpus: Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        """
        results: Dict[int, Dict[str, Union[str, List[str]]]] = {}
        eos_token_id = self.model.config.eos_token_id
        pad_token_id = self.model.config.pad_token_id
        if not pad_token_id:
            pad_token_id = eos_token_id
            self.model.config.pad_token_id = pad_token_id

        # Default parameters for each strategy
        default_params = {
            "generate": {
                "max_length": 20,  # Maximum length the generated tokens can have
                "max_new_tokens": None,  # Maximum number of tokens to generate, ignoring prompt tokens
                "min_length": 0,  # Minimum length of the sequence to be generated
                "min_new_tokens": None,  # Minimum number of tokens to generate, ignoring prompt tokens
                "early_stopping": False,  # Stopping condition for beam-based methods
                "max_time": None,  # Maximum time allowed for computation in seconds
                "do_sample": False,  # Whether to use sampling for generation
                "num_beams": 1,  # Number of beams for beam search
                "num_beam_groups": 1,  # Number of groups for beam search to ensure diversity
                "penalty_alpha": None,  # Balances model confidence and degeneration penalty in contrastive search
                "use_cache": True,  # Whether the model should use past key/values attentions to speed up decoding
                "temperature": 1.0,  # Modulates next token probabilities
                "top_k": 50,  # Number of highest probability tokens to keep for top-k-filtering
                "top_p": 1.0,  # Smallest set of most probable tokens with cumulative probability >= top_p
                "typical_p": 1.0,  # Conditional probability of predicting a target token next
                "epsilon_cutoff": 0.0,  # Tokens with a conditional probability > epsilon_cutoff will be sampled
                "eta_cutoff": 0.0,  # Eta sampling, a hybrid of locally typical sampling and epsilon sampling
                "diversity_penalty": 0.0,  # Penalty subtracted from a beam's score if it generates a token same as any other group
                "repetition_penalty": 1.0,  # Penalty for repetition of ngrams
                "encoder_repetition_penalty": 1.0,  # Penalty on sequences not in the original input
                "length_penalty": 1.0,  # Exponential penalty to the length for beam-based generation
                "no_repeat_ngram_size": 0,  # All ngrams of this size can only occur once
                "bad_words_ids": None,  # List of token ids that are not allowed to be generated
                "force_words_ids": None,  # List of token ids that must be generated
                "renormalize_logits": False,  # Renormalize the logits after applying all logits processors
                "constraints": None,  # Custom constraints for generation
                "forced_bos_token_id": None,  # Token ID to force as the first generated token
                "forced_eos_token_id": None,  # Token ID to force as the last generated token
                "remove_invalid_values": False,  # Remove possible NaN and inf outputs
                "exponential_decay_length_penalty": None,  # Exponentially increasing length penalty after a certain number of tokens
                "suppress_tokens": None,  # Tokens that will be suppressed during generation
                "begin_suppress_tokens": None,  # Tokens that will be suppressed at the beginning of generation
                "forced_decoder_ids": None,  # Mapping from generation indices to token indices that will be forced
                "sequence_bias": None,  # Maps a sequence of tokens to its bias term
                "guidance_scale": None,  # Guidance scale for classifier free guidance (CFG)
                "low_memory": None,  # Switch to sequential topk for contrastive search to reduce peak memory
                "num_return_sequences": 1,  # Number of independently computed returned sequences for each batch element
                "output_attentions": False,  # Whether to return the attentions tensors of all layers
                "output_hidden_states": False,  # Whether to return the hidden states of all layers
                "output_scores": False,  # Whether to return the prediction scores
                "return_dict_in_generate": False,  # Whether to return a ModelOutput instead of a plain tuple
                "pad_token_id": None,  # The id of the padding token
                "bos_token_id": None,  # The id of the beginning-of-sequence token
                "eos_token_id": None,  # The id of the end-of-sequence token
            },
            "greedy_search": {
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "eos_token_id": eos_token_id,  # End-of-sequence token ID
                "pad_token_id": pad_token_id,  # Padding token ID
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "contrastive_search": {
                "top_k": 1,  # Size of the candidate set for re-ranking in contrastive search
                "penalty_alpha": 0,  # Degeneration penalty; active when larger than 0
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
                "sequential": False,  # Switch to sequential topk hidden state computation to reduce memory if True
            },
            "sample": {
                "do_sample": True,  # Use sampling for generation if True
                "temperature": 0.6,  # Temperature for sampling
                "top_k": 50,  # Number of highest probability tokens to keep for top-k-filtering
                "top_p": 0.9,  # Cumulative probability for top-p-filtering
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "beam_search": {
                "num_beams": 4,  # Number of beams for beam search
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "beam_sample": {
                "num_beams": 4,  # Number of beams for beam search
                "temperature": 0.6,  # Temperature for sampling
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "group_beam_search": {
                "num_beams": 4,  # Number of beams for beam search
                "diversity_penalty": 0.5,  # Penalty for reducing similarity across different beam groups
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "constrained_beam_search": {
                "num_beams": 4,  # Number of beams for beam search
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "constraints": None,  # List of constraints to apply during beam search
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            },
        }

        # Merge default params with user-provided params
        strategy_params = {**default_params.get(decoding_strategy, {})}
        for k, v in generation_params.items():
            if k in strategy_params:
                strategy_params[k] = v

        # Prepare LogitsProcessorList and BeamSearchScorer for beam search strategies
        if decoding_strategy in ["beam_search", "beam_sample", "group_beam_search"]:
            logits_processor = LogitsProcessorList(
                [MinLengthLogitsProcessor(min_length=strategy_params.get("min_length", 0), eos_token_id=eos_token_id)]
            )
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                max_length=strategy_params.get("max_length", 20),
                num_beams=strategy_params.get("num_beams", 1),
                device=self.model.device,
                length_penalty=strategy_params.get("length_penalty", 1.0),
                do_early_stopping=strategy_params.get("early_stopping", False),
            )
            strategy_params.update({"logits_processor": logits_processor, "beam_scorer": beam_scorer})

            if decoding_strategy == "beam_sample":
                strategy_params.update({"logits_warper": LogitsProcessorList()})

        # Map of decoding strategy to method
        strategy_to_method = {
            "generate": self.model.generate,
            "greedy_search": self.model.greedy_search,
            "contrastive_search": self.model.contrastive_search,
            "sample": self.model.sample,
            "beam_search": self.model.beam_search,
            "beam_sample": self.model.beam_sample,
            "group_beam_search": self.model.group_beam_search,
            "constrained_beam_search": self.model.constrained_beam_search,
        }

        try:
            self.log.debug(f"Generating completion for prompt {prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            input_ids = input_ids.to(self.model.device)

            # Replicate input_ids for beam search
            if decoding_strategy in ["beam_search", "beam_sample", "group_beam_search"]:
                num_beams = strategy_params.get("num_beams", 1)
                input_ids = input_ids.repeat(num_beams, 1)

            # Use the specified decoding strategy
            decoding_method = strategy_to_method.get(decoding_strategy, self.model.generate)
            generated_ids = decoding_method(input_ids, **strategy_params)

            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            self.log.debug(f"Generated text: {generated_text}")

            return generated_text

        except Exception as e:
            self.log.exception(f"An error occurred: {e}")
            raise

    def load_models(
        self,
        model_name: str,
        processor_name: str,
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
            processor = FeatureExtractorClass.from_pretrained(
                os.path.join(self.input.get(), "/model"), torch_dtype=torch_dtype
            )
        else:
            processor = FeatureExtractorClass.from_pretrained(
                processor_name, revision=processor_revision, torch_dtype=torch_dtype
            )

        ModelClass = getattr(transformers, model_class)
        if quantization == 8:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    os.path.join(self.input.get(), "/model"),
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
                    os.path.join(self.input.get(), "/model"),
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
                    os.path.join(self.input.get(), "/model"),
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

    def done(self):
        """
        Finalizes the AudioBulk processing. Sends notification email if configured.

        This method should be called after all audio processing tasks are complete.
        It handles any final steps such as sending notifications or cleaning up resources.
        """
        if self.notification_email:
            self.output.flush()
            send_email(recipient=self.notification_email, bucket_name=self.output.bucket, prefix=self.output.s3_folder)
