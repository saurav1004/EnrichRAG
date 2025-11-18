import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class LLMAgent:
    def __init__(self, model_path, max_input_len=4096):
        self.llm = LLM(model=model_path,
                       tensor_parallel_size=1, # Change this if using >1 GPU for the LLM
                       trust_remote_code=True
                      )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_input_len = max_input_len

    def _truncate_prompt(self, prompt_text):
        """Truncates prompt from the middle to fit max_input_len."""
        tokens = self.tokenizer.encode(prompt_text)
        if len(tokens) <= self.max_input_len:
            return prompt_text
        
        print(f"Warning: Prompt truncated. Original length {len(tokens)}")
        # Truncate from the middle
        half_len = self.max_input_len // 2
        truncated_tokens = tokens[:half_len - 50] + tokens[-half_len + 50:]
        return self.tokenizer.decode(truncated_tokens)

    def generate(self, prompt, max_tokens=100, stop_tokens=None):
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=stop_tokens or [],
            include_stop_str_in_output=False
        )
        safe_prompt = self._truncate_prompt(prompt)
        output = self.llm.generate([safe_prompt], params, use_tqdm=False)
        return output[0].outputs[0].text

    def get_perplexity(self, text):
        safe_text = self._truncate_prompt(text)
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        output = self.llm.generate([safe_text], params, use_tqdm=False)
        
        if not output or not output[0].prompt_logprobs:
            print("Warning: Could not get logprobs, returning high perplexity.")
            return float('inf')

        prompt_logprobs = output[0].prompt_logprobs
        logprob_sum = 0.0
        token_count = 0
        
        for i in range(1, len(prompt_logprobs)):
            token_id = list(prompt_logprobs[i].keys())[0]
            logprob = prompt_logprobs[i][token_id]
            logprob_sum += logprob.logprob
            token_count += 1
            
        if token_count == 0:
            return float('inf')

        avg_logprob = logprob_sum / token_count
        perplexity = torch.exp(-torch.tensor(avg_logprob)).item()
        
        return perplexity