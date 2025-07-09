# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Mapping, Optional

from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Device, collect_from_async_generator, random_uuid

logger = init_logger(__name__)


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool:
        ...

    @property
    @abstractmethod
    def errored(self) -> bool:
        ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException:
        ...

    @abstractmethod
    def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            # Handle encoder-decoder beam search
            async for result in self._encoder_decoder_beam_search(
                prompt, request_id, params, preprocessor, tokenizer
            ):
                yield result
            return
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []

        for _ in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]

            tasks = []

            request_id = f"beam_search-{random_uuid()}"
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params,
                                      request_id_item)))
                tasks.append(task)

            output = await asyncio.gather(*tasks)

            output = [x[0] for x in output]

            new_beams = []
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        if token_id == tokenizer.eos_token_id and \
                            not ignore_eos:
                            completed.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens +
                                    [token_id] if include_stop_str_in_output
                                    else current_beam.tokens,
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    finish_reason="stop",
                                    stop_reason=tokenizer.eos_token_id))
                        else:
                            new_beams.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    multi_modal_data=current_beam.
                                    multi_modal_data,
                                    mm_processor_kwargs=current_beam.
                                    mm_processor_kwargs))

            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            all_beams = sorted_beams[:beam_width]

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output

    async def _encoder_decoder_beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
        preprocessor,
        tokenizer,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Beam search implementation for encoder-decoder models.
        
        For encoder-decoder models like Whisper:
        1. Process encoder input (e.g., audio features) once
        2. Run beam search on decoder side with shared encoder states
        3. Each beam maintains its own decoder token sequence
        """
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        # Process the encoder-decoder prompt
        processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)
        
        # For encoder-decoder models, processed_inputs contains both encoder and decoder info
        encoder_inputs = processed_inputs.get("encoder", {})
        decoder_inputs = processed_inputs.get("decoder", {})
        
        # Get the initial decoder prompt tokens
        decoder_prompt_token_ids = decoder_inputs.get("prompt_token_ids", [])
        prompt_text = decoder_inputs.get("prompt", "")
        
        # Get multimodal data (for models like Whisper with audio input)
        multi_modal_data = encoder_inputs.get("multi_modal_data") or decoder_inputs.get("multi_modal_data")
        mm_processor_kwargs = encoder_inputs.get("mm_processor_kwargs") or decoder_inputs.get("mm_processor_kwargs")

        tokenized_length = len(decoder_prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        # Create beam search parameters for single token generation
        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
        )
        
        # Initialize beams with the decoder prompt
        all_beams = [
            BeamSearchSequence(
                tokens=decoder_prompt_token_ids,
                cum_logprob=0,
                logprobs=[],
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs
            )
        ]
        completed = []

        # Run beam search for max_tokens steps
        for step in range(max_tokens):
            if not all_beams:
                break
                
            # Create prompts for this beam search step
            # For encoder-decoder models, we need to reconstruct the full prompt structure
            prompts_batch = []
            for beam in all_beams:
                # Reconstruct encoder-decoder prompt with updated decoder tokens
                enc_dec_prompt = {
                    "encoder_prompt": prompt["encoder_prompt"],  # Keep original encoder prompt
                    "decoder_prompt": TokensPrompt(
                        prompt_token_ids=beam.tokens,
                        multi_modal_data=beam.multi_modal_data,
                        mm_processor_kwargs=beam.mm_processor_kwargs
                    )
                }
                prompts_batch.append(enc_dec_prompt)

            # Generate next tokens for all beams
            tasks = []
            step_request_id = f"beam_search-{random_uuid()}"
            
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{step_request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params, request_id_item)
                    )
                )
                tasks.append(task)

            # Wait for all generations to complete
            outputs = await asyncio.gather(*tasks)
            outputs = [x[0] for x in outputs]

            # Process outputs and create new beams
            new_beams = []
            for i, current_beam in enumerate(all_beams):
                result = outputs[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        # Check if this is an EOS token
                        if token_id == tokenizer.eos_token_id and not ignore_eos:
                            # Beam is complete
                            final_tokens = (current_beam.tokens + [token_id] 
                                          if include_stop_str_in_output 
                                          else current_beam.tokens)
                            completed.append(
                                BeamSearchSequence(
                                    tokens=final_tokens,
                                    logprobs=current_beam.logprobs + [logprobs],
                                    cum_logprob=current_beam.cum_logprob + logprob_obj.logprob,
                                    finish_reason="stop",
                                    stop_reason=tokenizer.eos_token_id,
                                    multi_modal_data=current_beam.multi_modal_data,
                                    mm_processor_kwargs=current_beam.mm_processor_kwargs
                                )
                            )
                        else:
                            # Continue this beam
                            new_beams.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs + [logprobs],
                                    cum_logprob=current_beam.cum_logprob + logprob_obj.logprob,
                                    multi_modal_data=current_beam.multi_modal_data,
                                    mm_processor_kwargs=current_beam.mm_processor_kwargs
                                )
                            )

            # Sort beams and keep top beam_width
            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            all_beams = sorted_beams[:beam_width]

        # Add any remaining beams to completed
        completed.extend(all_beams)
        
        # Sort completed beams and get the best ones
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        # Decode the generated tokens for each beam
        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        # Create the final output
        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(
                    text=beam.text,
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens[tokenized_length:],
                    index=i,
                    logprobs=beam.logprobs,
                    finish_reason=beam.finish_reason if beam.finish_reason is not None else "length",
                    stop_reason=beam.stop_reason
                )
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=decoder_prompt_token_ids,
            prompt_logprobs=None
        )

        yield beam_search_output

    @abstractmethod
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """
        ...

    @abstractmethod
    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        ...

    @abstractmethod
    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool:
        ...

    @abstractmethod
    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        """Reset the prefix cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> None:
        """Load a new LoRA adapter into the engine for future requests."""
        ...
