#Most of the code here is taken from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_from_disk
from unsloth import is_bfloat16_supported

PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer
import legal_reasoning.utils as lru
import legal_reasoning.definitions as lrd

def main():

    model_name = "modelos/external/unsloth_Qwen2.5-3B-Instruct-bnb-4bit"
    output_dir="modelos/local/unsloth_Qwen2.5-3B-Instruct-bnb-4bit"
    run_name="unsloth_Qwen2.5-3B-Instruct-bnb-4bit-gsm8k"

    max_seq_length = 1024 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        #model_name = "Qwen/Qwen2.5-3B-Instruct",
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    dataset = lru.get_gsm8k_questions(load_from_disk("data/gsm8k"),split="train")
    dataset_test = lru.get_gsm8k_questions(load_from_disk("data/gsm8k"),split="test")

    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        run_name=run_name,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 8, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 200,
        num_train_epochs = 1, # Set to 1 for a full training run
        #max_steps = 250,
        save_steps = 250,
        eval_strategy ="steps",
        eval_steps=100,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = output_dir,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            lru.xmlcount_reward_func,
            lru.soft_format_reward_func,
            lru.strict_format_reward_func,
            lru.int_reward_func,
            lru.correctness_reward_func
        ],
        args = training_args,
        train_dataset=dataset,
        eval_dataset=dataset_test
    )
    trainer.train()

    model.save_lora(training_args.output_dir)

    return None

if __name__ == "__main__":
    main()