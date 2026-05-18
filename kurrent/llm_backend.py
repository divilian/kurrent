# Classes that encapsulate various kinds of LLM backends.

from typing import Optional
import subprocess
import tempfile


class LLMBackend:
    LLAMA_DIR = "/home/stephen/local/llama.cpp"
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

class OpenAIBackend(LLMBackend):
    def __init__(self, client, model: str="gpt-3.5-turbo"):
        self.client = client
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

class LocalLlamaBackend:
    """
    LLM backend that calls llama.cpp via the llama-cli binary.
    """
    def __init__(
        self,
        model_path: str | Path,
        llama_cli: str | Path = "llama-cli",
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        temperature: float = 0.2,
        max_tokens: int = 400,
        extra_args: Optional[list[str]] = None,
    ):
        self.model_path = str(model_path)
        self.llama_cli = str(llama_cli)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_args = extra_args or []

    def _extract_answer(self, output: str) -> str:
        print(f"Extracting from: {output}")
        lines = output.splitlines()
        # find the last prompt marker
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith(">"):
                return "\n".join(lines[i+1:]).strip()
        return output.strip()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]\n"

        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(prompt)
            prompt_path = f.name

        cmd = [
            self.llama_cli,
            "-m", self.model_path,
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.n_ctx),
            "--temp", str(self.temperature),
            "--n-predict", str(self.max_tokens),
            #"--simple-io",
            "--st",
            "--no-display-prompt",
            "--log-disable",
            "-f", prompt_path,
        ]

        if self.extra_args:
            cmd.extend(self.extra_args)

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.stderr.strip():
                print(f"Error from Llama! {result.stderr}\n")
                import sys ; sys.exit(1)
            print("|"+result.stdout.strip()+"|")
            return self._extract_answer(result.stdout.strip())
        finally:
            Path(prompt_path).unlink(missing_ok=True)

class LocalHFBackend(LLMBackend):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.2,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
