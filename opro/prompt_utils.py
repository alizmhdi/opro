# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompting GPT, Google Cloud, and local vLLM models."""

import time

try:
    import google.generativeai as palm
    _PALM_AVAILABLE = True
except ImportError:
    _PALM_AVAILABLE = False

try:
    import openai as _openai_legacy
    _LEGACY_OPENAI = True
except ImportError:
    _LEGACY_OPENAI = False

# openai >= 1.0 client (used for vLLM and modern OpenAI API)
try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_V1_AVAILABLE = True
except ImportError:
    _OPENAI_V1_AVAILABLE = False


def call_openai_server_single_prompt(
    prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with an input string."""
  import openai
  try:
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_decode_steps,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

  except openai.error.Timeout as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIConnectionError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API connection error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.ServiceUnavailableError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Service unavailable. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except OSError as e:
    retry_time = 5  # Adjust the retry time as needed
    print(
        f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
    )
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with a list of input strings."""
  if isinstance(inputs, str):
    inputs = [inputs]
  outputs = []
  for input_str in inputs:
    output = call_openai_server_single_prompt(
        input_str,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature,
    )
    outputs.append(output)
  return outputs


def call_palm_server_from_cloud(
    input_text, model="text-bison-001", max_decode_steps=20, temperature=0.8
):
  """Calling the text-bison model from Cloud API."""
  assert isinstance(input_text, str)
  assert model == "text-bison-001"
  all_model_names = [
      m
      for m in palm.list_models()
      if "generateText" in m.supported_generation_methods
  ]
  model_name = all_model_names[0].name
  try:
    completion = palm.generate_text(
        model=model_name,
        prompt=input_text,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
    )
    output_text = completion.result
    return [output_text]
  except:  # pylint: disable=bare-except
    retry_time = 10  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_palm_server_from_cloud(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )


# ---------------------------------------------------------------------------
# Local vLLM server (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def call_vllm_server_single_prompt(
    prompt,
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    max_decode_steps=1024,
    temperature=1.0,
    api_key="EMPTY",
    max_retries=5,
    retry_wait=5,
    n=1,
    client=None,
):
  """Call a locally served vLLM instance via its OpenAI-compatible HTTP API.

  Args:
    prompt: The user prompt string.
    base_url: Base URL of the vLLM server, e.g. ``http://localhost:8000/v1``.
    model: The model name as reported by the vLLM server (use the HuggingFace
      repo id or whatever name vLLM was started with).
    max_decode_steps: Maximum number of tokens to generate.
    temperature: Sampling temperature.
    api_key: Placeholder API key (vLLM ignores this but the client requires it).
    max_retries: How many times to retry on transient errors.
    retry_wait: Seconds to wait between retries.
    n: Number of completions to generate in a single request. When n > 1, all
      completions are batched on the server side (one HTTP round-trip).
    client: Optional pre-built ``openai.OpenAI`` client instance. When
      provided, it is reused instead of creating a new one (avoids repeated
      connection overhead).

  Returns:
    A single string when n=1, or a list of n strings when n > 1.
    Returns empty string(s) on persistent failure.
  """
  if not _OPENAI_V1_AVAILABLE:
    raise ImportError(
        "openai>=1.0 is required to call vLLM. "
        "Install it with: pip install 'openai>=1.0'"
    )

  _client = client if client is not None else _OpenAIClient(base_url=base_url, api_key=api_key)

  for attempt in range(max_retries):
    try:
      completion = _client.chat.completions.create(
          model=model,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=max_decode_steps,
          temperature=temperature,
          n=n,
      )
      if n == 1:
        return completion.choices[0].message.content
      return [choice.message.content for choice in completion.choices]
    except Exception as e:  # pylint: disable=broad-except
      print(
          f"[vLLM] Attempt {attempt + 1}/{max_retries} failed: {e}. "
          f"Retrying in {retry_wait}s..."
      )
      time.sleep(retry_wait)

  print("[vLLM] All retries exhausted. Returning empty string.")
  return "" if n == 1 else [""] * n


def call_vllm_server_func(
    inputs,
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    max_decode_steps=1024,
    temperature=1.0,
    api_key="EMPTY",
):
  """Call a locally served vLLM instance for a list of prompt strings.

  Args:
    inputs: A single prompt string or a list of prompt strings.
    base_url: Base URL of the vLLM server.
    model: The model name as served by vLLM.
    max_decode_steps: Maximum tokens to generate per prompt.
    temperature: Sampling temperature.
    api_key: Placeholder API key.

  Returns:
    A list of generated text strings, one per input prompt.
  """
  if isinstance(inputs, str):
    inputs = [inputs]
  outputs = []
  for prompt in inputs:
    output = call_vllm_server_single_prompt(
        prompt,
        base_url=base_url,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature,
        api_key=api_key,
    )
    outputs.append(output)
  return outputs
