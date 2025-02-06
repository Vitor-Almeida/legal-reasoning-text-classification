# legal-reasoning-text-classification
Applying GPRO Methods to Train Small LLMs for Complex Text Classification Tasks

Most of the code here is taken from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb and here https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=ptqkXK2D4d6p

Setting up enviorment:

pyenv uninstall legal-reasoning-text-classification
pyenv virtualenv 3.12.9 legal-reasoning-text-classification

pip install --upgrade pip
pip install --upgrade pillow
pip install unsloth vllm
pip install git+https://github.com/huggingface/trl.git
pip install wandb
