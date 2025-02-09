#Most of the code here is taken from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
import legal_reasoning.utils as lru
import legal_reasoning.definitions as lrd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load and prep dataset

def main():

    #model_name = "modelos/external/meta-llama_Llama-3.2-1B-Instruct"
    #output_dir="modelos/local/meta-llama_Llama-3.2-1B-Instruct"
    #run_name="meta-llama_Llama-3.2-1B-Instruct-gsm8k"

    #model_name = "modelos/external/meta-llama_Llama-3.2-3B-Instruct"
    #output_dir="modelos/local/meta-llama_Llama-3.2-3B-Instruct"
    #run_name="meta-llama_Llama-3.2-3B-Instruct-gsm8k"

    model_name = "modelos/external/Qwen_Qwen2.5-1.5B-Instruct"
    output_dir="modelos/local/Qwen_Qwen2.5-1.5B-Instruct"
    run_name="Qwen_Qwen2.5-1.5B-Instruct-gsm8k"

    #model_name = "modelos/external/Qwen_Qwen2.5-3B-Instruct"
    #output_dir="modelos/local/Qwen_Qwen2.5-3B-Instruct"
    #run_name="Qwen_Qwen2.5-3B-Instruct-gsm8k"

    #model_name = "modelos/external/unsloth_Qwen2.5-3B-Instruct-bnb-4bit"
    #output_dir="modelos/local/unsloth_Qwen2.5-3B-Instruct-bnb-4bit"
    #run_name="unsloth_Qwen2.5-3B-Instruct-bnb-4bit-gsm8k"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    max_prompt_length=1024
    max_completion_length=512

    dataset = lru.get_gsm8k_questions(load_from_disk("data/gsm8k"),split="train")
    dataset_test = lru.get_gsm8k_questions(load_from_disk("data/gsm8k"),split="test")

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        optim = "paged_adamw_8bit",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=6,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=1,
        #max_steps = 10,
        save_steps=100,
        save_total_limit=3,
        #eval_strategy ="steps",
        #eval_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        #vllm_device = "cuda:0",
        vllm_gpu_memory_utilization = 0.3,
        eval_accumulation_steps = 1,
        vllm_max_model_len = max_prompt_length + max_completion_length,
        # "enable_chunked_prefill": True,
        # "max_num_batched_tokens": 2048,
        #vllm_device="cuda:1",
        report_to="wandb" #I'm disabling Wandb.
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_8bit=True
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
        #tentar colocar flash attention aq
        #peft_config=peft_config
    )

    #model = prepare_model_for_kbit_training(model)
    #model = get_peft_model(model, peft_config)  # Add this line

    #model = AutoModelForCausalLM.from_pretrained(
    #    model_name,
    #    quantization_config=bnb_config,
    #    attn_implementation="flash_attention_2",
    #    device_map="auto"
    #)

    #lora_config = LoraConfig(
    #    task_type="CAUSAL_LM",
    #    r=8,
    #    lora_alpha=32,
    #    lora_dropout=0.1,
    #    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    #        "gate_proj", "up_proj", "down_proj"],
    #)

    #model = get_peft_model(model, lora_config)
    #model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            lru.xmlcount_reward_func,
            lru.soft_format_reward_func,
            lru.strict_format_reward_func,
            lru.int_reward_func,
            lru.correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        #eval_dataset=dataset_test
        #peft_config=peft_config
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    return None


if __name__ == "__main__":
    main()