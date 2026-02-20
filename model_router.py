"""Model routing for English/Dutch generation."""

import threading


class ModelRouter:
    """Routes requests to TinyLlama (English) or Qwen (Dutch)."""

    def __init__(self, tinyllama_path: str = './models/tinyllama', qwen_path: str = './models2/qwen'):
        self.tinyllama_path = tinyllama_path
        self.qwen_path = qwen_path

        self.tinyllama_model = None
        self.tinyllama_tokenizer = None
        self.qwen_model = None
        self.qwen_tokenizer = None

        self.transformers_available = False
        self.load_error = None
        self._init_transformers()

    def _init_transformers(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        except Exception as exc:
            self.load_error = f"transformers unavailable: {exc}"
            return

        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.TextIteratorStreamer = TextIteratorStreamer
        self.transformers_available = True

    def load_models(self):
        """Load both models (safe fallback if unavailable)."""
        if not self.transformers_available:
            print(f"[ROUTER] Running in fallback mode: {self.load_error}")
            return

        try:
            print('Loading TinyLlama 1.1B (English)...')
            self.tinyllama_tokenizer = self.AutoTokenizer.from_pretrained(
                self.tinyllama_path, local_files_only=True, trust_remote_code=True
            )
            self.tinyllama_model = self.AutoModelForCausalLM.from_pretrained(
                self.tinyllama_path,
                local_files_only=True,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype='auto',
            )
            self.tinyllama_model.eval()
            print('✓ TinyLlama loaded')
        except Exception as exc:
            self.load_error = f'TinyLlama load failed: {exc}'
            self.tinyllama_model = None
            self.tinyllama_tokenizer = None
            print(f'[ROUTER] {self.load_error}')

        try:
            print('Loading Qwen 1.5B (English + Dutch)...')
            self.qwen_tokenizer = self.AutoTokenizer.from_pretrained(
                self.qwen_path, local_files_only=True, trust_remote_code=True
            )
            self.qwen_model = self.AutoModelForCausalLM.from_pretrained(
                self.qwen_path,
                local_files_only=True,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype='auto',
            )
            self.qwen_model.eval()
            print('✓ Qwen loaded')
        except Exception as exc:
            self.load_error = f'Qwen load failed: {exc}'
            self.qwen_model = None
            self.qwen_tokenizer = None
            print(f'[ROUTER] {self.load_error}')

    def route_and_generate(self, prompt: str, language: str, max_tokens: int = 400):
        """Yield generated tokens for selected language route."""
        if language == 'Dutch':
            if self.qwen_model and self.qwen_tokenizer:
                yield from self._generate_with_model(self.qwen_model, self.qwen_tokenizer, prompt, max_tokens)
                return
        else:
            if self.tinyllama_model and self.tinyllama_tokenizer:
                yield from self._generate_with_model(self.tinyllama_model, self.tinyllama_tokenizer, prompt, max_tokens)
                return

        fallback = (
            "Model unavailable. Please provide local model files for TinyLlama/Qwen "
            "or adjust model paths in model_router.py."
        )
        for token in fallback.split(' '):
            yield token + ' '

    def _generate_with_model(self, model, tokenizer, prompt: str, max_tokens: int):
        messages = [
            {'role': 'system', 'content': 'You are a helpful repair assistant.'},
            {'role': 'user', 'content': prompt},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([input_text], return_tensors='pt').to(model.device)
        streamer = self.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **model_inputs,
            'max_new_tokens': max_tokens,
            'temperature': 0.3,
            'do_sample': True,
            'streamer': streamer,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.eos_token_id,
        }

        worker = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        worker.start()
        for token in streamer:
            yield token
        worker.join()
