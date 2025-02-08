# legal-reasoning-text-classification
Applying GPRO Methods to Train Small LLMs for Complex Text Classification Tasks

Most of the code here is taken from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb and here https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=ptqkXK2D4d6p

Setting up enviorment:

This was trained using 2x 3090

pyenv uninstall legal-reasoning-text-classification
pyenv virtualenv 3.12.9 legal-reasoning-text-classification

pip install --upgrade pip
pip install --upgrade pillow
pip install vllm
pip install git+https://github.com/huggingface/trl.git
pip install wandb
pip install bitsandbytes



#com unsloth:

#pip install --upgrade pip
#pip install unsloth vllm
#pip install --upgrade pillow
#pip install diffusers
#pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
#pip install wandb