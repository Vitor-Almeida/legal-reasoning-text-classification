from datasets import load_from_disk

import legal_reasoning.utils as lru
import legal_reasoning.definitions as lrd

from vllm import SamplingParams, LLM
from vllm import LLM, SamplingParams

dataset = lru.get_gsm8k_questions(load_from_disk("data/gsm8k"),split="train")
dataset_test = lru.get_gsm8k_questions(load_from_disk("data/gsm8k"),split="test")

#llm = LLM(model="modelos/external/meta-llama_Llama-3.2-1B-Instruct")
llm = LLM(model="modelos/local/meta-llama_Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)

conversation = [
    {"role" : "system", "content" : lrd.SYSTEM_PROMPT},
    {"role" : "user", "content" : dataset_test['question'][0]}
]

#dataset_test['answer'][0]

outputs = llm.chat(conversation, sampling_params=sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(output)