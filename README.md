# Direct Consistency Optimization for Compositional Text-to-Image Personalization
This is an official implementation of paper 'Direct Consistency Optimization for Compositional Text-to-Image Personalization' 
- [paper](https://arxiv.org/abs/2402.12004) 
- [project page](https://dco-t2i.github.io/) 

Our code is based on [diffusers](https://github.com/huggingface/diffusers), which we fine-tune [SDXL](https://huggingface.co/docs/diffusers/using-diffusers/sdxl) using LoRA from [peft](https://github.com/huggingface/peft) library. 

## Installation
We recommend to install from the source the latest version of diffusers:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then go to the repository and install via
```bash
cd dco/
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups. 
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

## Subject Personalization 
### Data preparation

We encourage to use **comprehensive caption** for text-to-image personlization, which provides descriptive visual details on the attributes, backgrounds, etc. Also we do not use rare token identifier (e.g., 'sks'), which may inherit the unfavorable semantics. We also train additional textual embeddings to enhance the subject fidelity. See paper for details.

In `dataset/dreambooth/dog/config.json`, we provide an example of comprehensive captions that we used:
```
'full_ti': {
    "images":[
        "dataset/dreambooth/dog/00.jpg",
        "dataset/dreambooth/dog/01.jpg",
        "dataset/dreambooth/dog/02.jpg", 
        "dataset/dreambooth/dog/03.jpg",
        "dataset/dreambooth/dog/04.jpg"
    ],
    "prompts": [
        "a closed-up photo of a <dog> in front of trees, macro style",
        "a low-angle photo of a <dog> sitting on a ledge in front of blossom trees, macro style",
        "a photo of a <dog> sitting on a ledge in front of red wall and tree, macro style",
        "a photo of side-view of a <dog> sitting on a ledge in front of red wall and tree, macro style",
        "a photo of a <dog> sitting on a street, in front of lush trees, macro style"
    ],
    "base_prompts": [
        "a closed-up photo of a dog in front of trees, macro style",
        "a low-angle photo of a dog sitting on a ledge in front of blossom trees, macro style",
        "a photo of a dog sitting on a ledge in front of red wall and tree, macro style",
        "a photo of side-view of a dog sitting on a ledge in front of red wall and tree, macro style",
        "a photo of a dog sitting on a street, in front of lush trees, macro style"
        ],
    "inserting_tokens" : ["<dog>"],
    "initializer_tokens" : ["dog"]
}
```
`images` is a list of directories for training images, `prompts` are list of training prompts with training tokens (*e.g.,* `<dog>`), and `base_prompts` are list of training prompts without new tokens. `inserting tokens` are list of learning tokens, and `initializer_tokens` are list of tokens that are used for initialization. If you do not want initializer token than put empty string (*i.e.,* `""`) in `initializer_tokens`. Note that the norm of token embeddings are rescaled after each iteration to be same as original one.


### Training scripts
To train the model, run following command:
```
accelerate launch customize.py \
    --config_dir="dataset/dreambooth/dog/config.json" \
    --config_name="full_ti" \
    --output_dir="./output" \
    --learning_rate=5e-5 \
    --text_encoder_lr=5e-6 \
    --dcoloss_beta=1000 \
    --rank=32 \
    --max_train_steps=2000 \
    --checkpointing_steps=1000 \
    --seed="0" \
    --train_text_encoder_ti
```
Note that `--dcoloss_beta` is a hyperparameter that is used for DCO loss (1000-2000 works fine in our experiments). `--train_text_encoder_ti` is to indicate learning with textual embeddings. 

### Inference
To infer with reward guidance, import `RGPipe` from `reward_guidance.py`. Then load lora weights and textual embeddings:
```
import torch
import os
from safetensors.torch import load_file
from reward_guidance import RGPipe

pipe = RGPipe.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")    
lora_dir = "OUTPUT_DIR" # saved lora directory
pipe.load_lora_weights(lora_dir)

inserting_tokens = ["<dog>"] # load new tokens    
state_dict = load_file(lora_dir+"/learned_embeds.safetensors")
pipe.load_textual_inversion(state_dict["clip_l"], token=inserting_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion(state_dict["clip_g"], token=inserting_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

prompt = "A <dog> playing saxophone in sticker style" # prompt including new tokens
base_prompt = "A dog playing saxophone in sticker style" # prompt without new tokens

seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

rg_scale = 3.0 # rg scale. 0.0 for original CFG sampling
if rg_scale > 0.0:
    image = pipe.my_gen(
        prompt=base_prompt,
        prompt_ti=prompt, 
        generator=generator,
        cross_attention_kwargs={"scale": 1.0},
        guidance_scale=7.5,
        guidance_scale_lora=rg_scale,
        ).images[0]
else:
    image = pipe(
        prompt=prompt, 
        generator=generator,
        cross_attention_kwargs={"scale": 1.0},
        guidance_scale=7.5,
        ).images[0]
image
```

## Style Personlization
### Data Preparation
We use same format as before, but we do not train textual embeddings for style personalization. The example config is given by 
```
"style":{
    "images" : ["dataset/styledrop/style.jpg"],
    "prompts": ["A person working on a laptop in flat cartoon illustration style"]
}
```

### Training scripts
```
accelerate launch customize.py \
    --config_dir="dataset/styledrop/config.json" \
    --config_name="style_1" \
    --output_dir="./output_style" \
    --learning_rate=5e-5 \
    --dcoloss_beta=1000 \
    --rank=64 \
    --max_train_steps=1000 \
    --seed="0" \
    --offset_noise=0.1
```
Note that we use `--offset_noise=0.1` to learn solid color of the style image.

The inference is same as above.

## My Subject in My Style
DCO fine-tuned models can be easily merged without any post-processing. Simply, add following codes during inference: 
```
pipe.load_lora_weights(subject_lora_dir, adapter_name="subject")
if args.text_encoder_ti:
    state_dict = load_file(subject_lora_dir+"/learned_embeds.safetensors")
    pipe.load_textual_inversion(state_dict["clip_l"], token=inserting_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    pipe.load_textual_inversion(state_dict["clip_g"], token=inserting_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

pipe.load_lora_weights(style_lora_dir, adapter_name="style")
pipe.set_adapters(["subject", "style"], adapter_weights=[1.0, 1.0])
```


## BibTex
```
@article{lee2024direct,
  title={Direct Consistency Optimization for Compositional Text-to-Image Personalization},
  author={Lee, Kyungmin and Kwak, Sangkyung and Sohn, Kihyuk and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2402.12004},
  year={2024}
}
``` 
