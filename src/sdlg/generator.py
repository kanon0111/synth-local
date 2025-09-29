"""
sdlg.generator
- ローカル推論（Transformers）でテキスト生成
- chat=True で Instruct/Chat テンプレを使用
- seed による再現性オプションをサポート
"""
from typing import List, Dict, Any, Optional
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextGenerator:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        chat: bool = False,
        system_text: str = "あなたは丁寧な日本語で答えるカスタマーサポート担当者です。",
        seed: Optional[int] = None,
    ) -> None:
        self.chat = chat
        self.system_text = system_text
        self.seed = seed

        # 乱数初期化（可能な範囲で再現性を高める）
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token is None:
            if self.tok.eos_token is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({"pad_token": "<|pad|>"})

        # Model（GPUがあれば自動で乗る）
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto"
        )

        # 語彙サイズ調整（pad_token追加時など）
        if self.model.get_input_embeddings().num_embeddings != len(self.tok):
            self.model.resize_token_embeddings(len(self.tok))

        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _generator(self, device):
        g = torch.Generator(device=device)
        if self.seed is not None:
            g.manual_seed(self.seed)
        return g

    # --- Free-form 続き書き ---
    def _gen_freeform(self, prompt: str) -> str:
        enc = self.tok(prompt, return_tensors="pt")
        dev = getattr(self.model, "device", next(self.model.parameters()).device)
        enc = {k: v.to(dev) for k, v in enc.items()}
        g = self._generator(dev)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
                generator=g
            )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

    # --- Chat（Instruct） ---
    def _gen_chat(self, user_text: str) -> str:
        msgs = [
            {"role": "system", "content": self.system_text},
            {"role": "user", "content": user_text},
        ]
        dev = getattr(self.model, "device", next(self.model.parameters()).device)
        input_ids = self.tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(dev)
        g = self._generator(dev)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tok.eos_token_id,
                generator=g
            )
        new_tokens = out[:, input_ids.shape[1]:]
        return self.tok.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    # --- Public API ---
    def generate(self, prompts: List[str], n_per_prompt: int = 1) -> List[Dict[str, Any]]:
        fn = self._gen_chat if self.chat else self._gen_freeform
        rows: List[Dict[str, Any]] = []
        for p in prompts:
            for _ in range(n_per_prompt):
                rows.append({"prompt": p, "response": fn(p)})
        return rows
