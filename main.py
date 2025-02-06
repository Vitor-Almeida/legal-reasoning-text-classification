#Most of the code here is taken from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
import legal_reasoning.utils as lru
import legal_reasoning.definitions as lrd
from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load and prep dataset

def main():

    model_name = "modelos/external/meta-llama_Llama-3.2-1B-Instruct"

    output_dir="modelos/local/meta-llama_Llama-3.2-1B-Instruct"
    run_name="meta-llama_Llama-3.2-1B-Instruct-gsm8k"

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
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=.6,
        vllm_device="cuda:1",
        report_to="wandb" #I'm disabling Wandb.
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda")

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        eval_dataset=dataset_test,
        #peft_config=peft_config
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    return None


if __name__ == "__main__":
    main()