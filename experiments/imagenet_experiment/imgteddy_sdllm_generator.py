import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from tgate import TgateSDDeepCacheLoader
from DeepCache import DeepCacheSDHelper
import subprocess
import re
import random
import numpy as np
import torch
import cv2
import csv
#import wandb
from torch import autocast
from PIL import Image
from torchvision import transforms
from collections import Counter
from llama_cpp import Llama
#from cifar10_classifier.model import VGGNet
#run =wandb.init(project="sinvadtestfitness2")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device = 'cuda')
torch.use_deterministic_algorithms(True)

def calculate_fitness(logit, label):

    expected_logit = logit[label]
    # Select the two best indices [perturbed logit]
    best_indices = np.argsort(-logit)[:2]
    best_index1, best_index2 = best_indices
    if best_index1 == label:
        best_but_not_expected = best_index2
    else:
        best_but_not_expected = best_index1
    new_logit = logit[best_but_not_expected]
    # Compute fitness value
    fitness = expected_logit - new_logit
    return fitness
proj_path = "./imgteddy_generatorlow/"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/perturb-result', exist_ok=True)
csv_path = os.path.join(proj_path, "used_prompts.csv")


# === Load Model and Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
generator = torch.Generator(device='cuda')

classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
classifier.eval()
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load SD
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    variant="fp16", torch_dtype=torch.float16, safety_checker=None
).to(device)
#pipe.load_lora_weights("./lora_weights/cifar10")
pipe.load_lora_weights("./lora_weights/teddy-weight", adapter_name="concept")

# Load second LoRA (style)
pipe.load_lora_weights("./lora_weights/teddysty-weight", adapter_name="style")

# Merge them into an active adapter
pipe.set_adapters(["concept", "style"], adapter_weights=[0.01,0.3])
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
pipe = TgateSDDeepCacheLoader(
       pipe,
       cache_interval=3,
       cache_branch_id=0,
).to(device)
saved_images = 0
seedSelect = 0
random.seed(seedSelect)
np.random.seed(seedSelect)
torch.manual_seed(seedSelect)
generator = generator.manual_seed(seedSelect)
height = 512
min_val = -5.26750659942627
max_val = 5.51045942306519
init_perturbation =0.00215559320449829                                           # 0.00215559320449829
perturbation_size =0.00107779660224915                                           #0.00107779660224915                                                                        
best_left = 10
gen_num = 250
pop_size = 25
fitness_scores = []
all_img_lst = []
imgs_to_samp = 100
image_info = []
predicted_labels = []
unique_prompts = set()
height = 512
width = 512
num_inference_steps = 25
for n in range(imgs_to_samp):
    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device).to(torch.float16)
    best_prompt = None
    best_score = -1
    best_image = None
    best_pred_label = None

    style_phrases = {
    "bed": "sit on a bed",
    "chair": "sit on a chair",
    "table": "sit on a table" }
    selected_style = random.choice(list(style_phrases.keys()))
    style_phrase = style_phrases[selected_style]
    expected_label=850
 
    instruction = f"""
You are a prompt generator for a text-to-image model.

Generate exactly one short prompt starting with:
A photo of teddy bear...

Rules:
- The prompt must be under 12 words.
- Use the style phrase provided in the variable: {style_phrase}
- The prompt format must be:
    A photo of teddy bear {style_phrase}
- Do not add any extra adjectives, objects, or background elements.
- Output only the final prompt — no explanations or additional text.

"""    
    
    prompt_text = (
         "<|start_header_id|>user<|end_header_id|>\n\n"
         f"{instruction}"
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    llm = Llama(model_path="/home/vincenzo.riccio/models/llama-3/Llama-3-8B-Instruct-32k-v0.1.Q4_K_M.gguf",seed = seedSelect + n) 
    
    response = llm(
        prompt = prompt_text,
        temperature=0.5,
        top_p=0.95,
        seed=seedSelect + n,
        stop=["\n"]
    )
    # Extract prompt
    if not response.get("choices") or not response["choices"][0].get("text"):
        print(f"[{expected_label}-{n}] No prompt generated!")
        continue

    raw_output = response["choices"][0]["text"].strip()
    lines = raw_output.splitlines()

    # Extract first valid prompt line
    prompt = next((line for line in lines if line.lower().startswith("a photo of")), "").strip()
    if not prompt:
       print(f"[{expected_label}-{n}] Invalid or empty prompt: {raw_output}")
       continue
    print(f"[{expected_label}-{n}] {prompt}")
    print("Raw output:", response)

    with open("prompt_debug.csv", "a") as f:
        f.write(f"{n},{seedSelect},{expected_label},{prompt}\n")

    # Minimal validation
    if not prompt.lower().startswith("a photo of"):
        print(f"[{expected_label}-{n}] Bad prompt format: {prompt}")
        continue

    if prompt not in unique_prompts:
        unique_prompts.add(prompt)
        print(prompt)

    with torch.inference_mode():
        init_img = pipe.tgate(prompt=prompt, guidance_scale=3.5, num_inference_steps=num_inference_steps, latents=original_lv)["images"][0]
    tensor_image = transform(init_img).unsqueeze(0).to(device)
    logits = classifier(tensor_image).squeeze().detach().cpu()
    original_label = np.argmax(logits).item()

    if original_label == expected_label:
       filename = f"image_{saved_images}_X{expected_label}_Y{original_label}.png"
       init_img.save(os.path.join(proj_path, filename))
       # only log saved images to this CSV
       with open(csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([filename, expected_label, prompt, original_label])
       print(f"Saved: {filename} | Prompt: {prompt}")
       init_pop = [
         original_lv + init_perturbation * torch.randn((pipe.unet.config.in_channels, height // 8, width // 8), device=device).to(torch.float16)
         for _ in range(pop_size)
       ]
       best_fitness_score = float('inf')
       best_image_tensor = None
       best_image_index = -1
       now_pop = init_pop
       prev_best = np.inf
       for g_idx in range(gen_num):
          # Flatten all tensors in now_pop into a single tensor for min/max evaluation
           indivs_lv = torch.cat(now_pop, dim=0).view(-1, 4, height // 8, width // 8).to(torch.float16)
           print(indivs_lv.shape)  
           with torch.inference_mode():                         
                perturb_img = pipe.tgate([prompt]*(pop_size),guidance_scale =1.4, generator= generator, 
                   num_inference_steps=num_inference_steps,
                   latents=indivs_lv,
                )["images"]
           # all_img_lst.append(perturb_img)
           torch.cuda.empty_cache()

           tensor_image2 =torch.stack([transform(image) for image in perturb_img])
           # tensor_image2 = transform(last_image)
           tensor_image2 = tensor_image2.to(device)
           all_logits = classifier(tensor_image2).detach().cpu().numpy()
           perturb_label1 = np.argmax(all_logits).item()
           print(all_logits.shape)
           print(tensor_image2.shape)
           os.makedirs(os.path.join(proj_path, 'generated_images'), exist_ok=True)
          
           fitness_scores = [
               calculate_fitness(all_logits[k_idx], original_label)
               for k_idx in range(pop_size)
           ]
           print("print fitness",len(fitness_scores))

           # Find the minimum fitness score in the current generation
           current_min_index = np.argmin(fitness_scores)
           current_min_fitness = fitness_scores[current_min_index]
    
           # Update best tracking variables if the current minimum is less than the tracked best
           if current_min_fitness < best_fitness_score:
              best_fitness_score = current_min_fitness
              best_image_tensor  = tensor_image2[current_min_index]  # Ensure a deep copy
              best_image_index = current_min_index
           # Perform selection
           selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True,
              )[-best_left:]
           now_best = np.min(fitness_scores)
           parent_pop = [now_pop[idx] for idx in selected_indices]
           print(parent_pop[-1].shape)
           print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
          # wandb.log({"ft_score":now_best})
           if now_best < 0:
              break
           elif now_best == prev_best:
               perturbation_size *= 2
           else:
               perturbation_size = init_perturbation
           k_pop = []
           # print("Size of parent_pop:", len(parent_pop))
           # select k-idx for cross_over genes
           for k_idx in range(pop_size - best_left):
               mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
              # print("mom_idx:", mom_idx, "pop_idx:", pop_idx)
               spl_idx = np.random.choice(4, size=1)[0]
               k_gene = torch.cat(
                   [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                   dim=1,
               )  # crossover
               # Mutation
               diffs = (k_gene != original_lv).float()
               k_gene += (
                      perturbation_size * torch.randn(k_gene.size(), device=k_gene.device) * diffs
               )  # random adding noise only to diff places
              # k_gene = torch.clamp(k_gene, min_val, max_val)
              # interp_mask = binom_sampler.sample()
              # k_gene = interp_mask * original_lv + (1 - interp_mask) * k_gene
               k_pop.append(k_gene)
               # print("Size of k_pop:", len(k_pop))

           # Combine parent_pop and k_pop for the next generation
           now_pop = parent_pop + k_pop
           prev_best = now_best
           # Apply a final clamp across all now_pop before next iteration starts
           now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

      
     # After the loop, save the last image if it exists
       if best_image_tensor is not None:
          tensor_image = best_image_tensor.to(device)
          tensor_image = tensor_image.unsqueeze(0).to(device)
          tensor_image_np= tensor_image.squeeze().detach().cpu().numpy()
          perturb_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
          perturb_label = np.argmax(perturb_logit).item()
          predicted_labels.append(perturb_label)
          all_img_lst.append(tensor_image_np)
          # Save the image as a numpy array
          print("Best image tensor shape:", best_image_tensor.shape)
          print("Tensor dtype before save:", tensor_image.dtype)
          print("Before normalization:")
          print("  Min:", best_image_tensor.min().item())
          print("  Max:", best_image_tensor.max().item())
          tensor_image = best_image_tensor.cpu().detach()
          # Normalize to [0, 1] range (visual saving)
          min_val1 = tensor_image.min()
          max_val1 = tensor_image.max()

          if max_val1 > min_val1:
             tensor_image = (tensor_image - min_val1) / (max_val1 - min_val1)
          else:
             tensor_image = torch.zeros_like(tensor_image)  # avoid divide-by-zero

          # Convert to uint8 [0, 255]
          tensor_image = (tensor_image * 255).clamp(0, 255).to(torch.uint8)

          best_image_pil = transforms.ToPILImage()(tensor_image)
          image_filename = f'image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{perturb_label}.png'
          last_image_path = os.path.join(proj_path, 'perturb-result', image_filename)
          best_image_pil.save(last_image_path)

          # Save the image as a numpy array
          np_filename = f'image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{perturb_label}'
          np.save(os.path.join(proj_path,'generated_images', np_filename), tensor_image_np)
       saved_images += 1
    else: 
       # IF no match, simply skip to the next iteration (no image is saved or processed)
       print(f"Label mismatch for image {saved_images}:expected {expected_label},got {original_label}")
# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(proj_path, "bound_imgpiz_sd.npy"), all_imgs)
