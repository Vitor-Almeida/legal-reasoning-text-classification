#Most of the code here is taken from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import legal_reasoning.utils
import legal_reasoning.definitions

# Load and prep dataset

def main():

    #utils.extract_xml_answer()
    print(definitions.SYSTEM_PROMPT)

    return None


if __name__ == "__main__":
    main()