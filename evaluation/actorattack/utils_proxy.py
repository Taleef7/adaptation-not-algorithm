import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CALL_SLEEP = int(os.getenv("ACTORATTACK_CALL_SLEEP_SECONDS", "1"))
clients = {}
_local_clients = {}
_local_clients_lock = threading.Lock()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _get_backend() -> str:
    return _env("ACTORATTACK_BACKEND", "hf_local").strip().lower()


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


class _CompletionResponse:
    def __init__(self, content: str):
        self.choices = [_Choice(message=_Message(content=content))]


class _LocalCompletionsAPI:
    def __init__(self, parent: "LocalHFClient"):
        self._parent = parent

    def create(self, model: str, messages: List[dict], temperature: float = 0, **kwargs):
        # Accept both OpenAI-standard max_tokens and HF-style max_new_tokens;
        # explicit value wins over default. response_format is silently ignored
        # (the local model generates free text; callers must parse it themselves).
        max_new_tokens = kwargs.get("max_new_tokens") or kwargs.get("max_tokens")
        content = self._parent.generate(
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return _CompletionResponse(content)


class _LocalChatAPI:
    def __init__(self, parent: "LocalHFClient"):
        self.completions = _LocalCompletionsAPI(parent)


class LocalHFClient:
    def __init__(self, model_path: str, model_name: str):
        self.model_path = str(Path(model_path))
        self.model_name = model_name
        self.default_max_new_tokens = int(_env("ACTORATTACK_LOCAL_MAX_NEW_TOKENS", "256"))
        self.default_temperature = float(_env("ACTORATTACK_LOCAL_TEMPERATURE", "0"))
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self.chat = _LocalChatAPI(self)

    def _load_model(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            dtype_name = _env("ACTORATTACK_LOCAL_DTYPE", "bfloat16").lower()
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()

    def _messages_to_prompt(self, messages: List[dict]) -> str:
        assert self._tokenizer is not None
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            return "\n".join(parts)

    def generate(self, messages: List[dict], temperature: float = 0, max_new_tokens: int = None) -> str:
        self._load_model()
        assert self._tokenizer is not None and self._model is not None
        import torch

        prompt = self._messages_to_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        use_temperature = self.default_temperature if temperature is None else float(temperature)
        do_sample = use_temperature > 0
        max_tokens = self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = use_temperature

        with self._generate_lock:
            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **generation_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


def _get_or_create_local_client(kind: str, model_path: str, model_name: str):
    key = f"{kind}:{model_path}"
    with _local_clients_lock:
        client = _local_clients.get(key)
        if client is None:
            client = LocalHFClient(model_path=model_path, model_name=model_name)
            _local_clients[key] = client
        return client


def initialize_clients():
    backend = _get_backend()

    if backend != "hf_local":
        raise RuntimeError(
            f"Unsupported ACTORATTACK_BACKEND='{backend}'. Only hf_local is supported."
        )

    target_model_name = _env("TARGET_MODEL_NAME", "target_model")
    attack_model_name = _env("ATTACK_MODEL_NAME", "attack_model")
    target_model_path = _env("ACTORATTACK_TARGET_MODEL_PATH")
    attack_model_path = _env("ACTORATTACK_ATTACKER_MODEL_PATH")

    if target_model_path:
        clients["target"] = _get_or_create_local_client("target", target_model_path, target_model_name)
    if attack_model_path:
        clients["attack"] = _get_or_create_local_client("attack", attack_model_path, attack_model_name)

    gpt_api_key = _env("OPENAI_API_KEY")
    if gpt_api_key:
        clients["gpt"] = OpenAI(api_key=gpt_api_key)


initialize_clients()


def get_client(model_name):
    target_model_name = _env("TARGET_MODEL_NAME", "target_model")
    attack_model_name = _env("ATTACK_MODEL_NAME", "attack_model")

    if model_name == target_model_name:
        return clients.get("target")
    if model_name == attack_model_name:
        return clients.get("attack")
    if "gpt" in model_name.lower():
        return clients.get("gpt")

    return clients.get("target")


def read_prompt_from_file(filename):
    with open(filename, "r") as file:
        return file.read()


def read_data_from_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


def parse_json(output):
    try:
        output = "".join(output.splitlines())
        if "{" in output and "}" in output:
            start = output.index("{")
            end = output.rindex("}")
            output = output[start : end + 1]
        return json.loads(output)
    except Exception as exc:
        print("parse_json:", exc)
        return None


def check_file(file_path):
    if os.path.exists(file_path):
        return file_path
    raise IOError(f"File not found error: {file_path}.")


def gpt_call(client, query: Union[List, str], model_name="gpt-4o", temperature=0):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "user", "content": query}]
    else:
        messages = [{"role": "user", "content": str(query)}]

    for _ in range(3):
        try:
            if client is None:
                raise RuntimeError(f"No client available for model: {model_name}")

            if "o1-" in model_name:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                )
            return completion.choices[0].message.content
        except Exception as exc:
            print(f"GPT_CALL Error: {model_name}:{exc}")
            time.sleep(CALL_SLEEP)
            continue
    return ""


def gpt_call_append(client, model_name, dialog_hist: List, query: str):
    dialog_hist.append({"role": "user", "content": query})
    resp = gpt_call(client, dialog_hist, model_name=model_name)
    dialog_hist.append({"role": "assistant", "content": resp})
    return resp, dialog_hist
