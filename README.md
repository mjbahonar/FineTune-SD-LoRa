
## Fine-Tuning Stable Diffusion with LoRA

I created this file using the following resources:

*   [Fine-Tuning Stable Diffusion with LoRA](https://machinelearningmastery.com/fine-tuning-stable-diffusion-with-lora) (Webpage)
*   [Fine-Tune Stable Diffusion with LoRA for as Low as $1](https://youtu.be/Zev6F0T1L3Y?t=458) (Video)
*   [Pokémon Dataset](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh) (Dataset)
*   [Diffusers GitHub Repository](https://github.com/huggingface/diffusers/) (GitHub)
*   [How to Fine-Tune with LoRA by Hugging Face](https://huggingface.co/docs/diffusers/en/training/lora) (Documentation)

### 1. Prepare the Dataset
In this guide, we will fine-tune a Stable Diffusion model using LoRA with the **[Pokémon dataset](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh)** from Hugging Face. This dataset can easily be replaced with another.

If you are creating your **own dataset**, you can prepare a CSV file named `metadata.csv`. The first column should contain `file_name`, and the second column should contain corresponding text captions. [[1](https://machinelearningmastery.com/fine-tuning-stable-diffusion-with-lora/)]

### 2. Set Up the Environment
Next, install the required libraries, including `diffusers`, `accelerate`, and `wandb`. [[1]](https://machinelearningmastery.com/fine-tuning-stable-diffusion-with-lora/). The main script used for this purpose can be found [here](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py).

### 3. Run the Training Script
Use the `accelerate launch` command with the training script, specifying the dataset, model, and hyperparameters. A detailed explanation of each parameter can be found [here](https://learnopencv.com/fine-tuning-stable-diffusion-3-5m/).

### 4. Training Process
The training process can take several hours, even with a high-end GPU.

### 5. Using Your Trained LoRA Model
After training, you will have a small weight file, typically named `pytorch_lora_weights.safetensors`. You can use it by loading it into a Stable Diffusion pipeline with:
`pipe.unet.load_attn_procs(model_path)`
