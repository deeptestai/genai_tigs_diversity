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
from celeb_classifier.celeb_resnet_model import AttributeClassifier
import sys
sys.path.append("/home/vincenzo.riccio/ddpm_diffuser/style-generator/compute-perturbation")
import celeb_sd_params
#run =wandb.init(project="sinvadtestfitness2")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device = 'cuda')
torch.use_deterministic_algorithms(True)

proj_path = "./celebahq_icst_generatorlow/"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/perturb-result', exist_ok=True)
csv_path = os.path.join(proj_path, "used_prompts.csv")


# === Load Model and Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
generator = torch.Generator(device='cuda')

attribute_list = ["smiling-face", "eye-glasses", "curtain-bangs"]
ATTR_CSV_PATH = "./CelebAHQ-attribute-label.txt"
attributes = []
with open(ATTR_CSV_PATH, "r") as f:
    n_images = int(f.readline().strip())         # first line = number of images
    attr_names = f.readline().strip().split()    # second line = 40 attribute names
    attributes = attr_names                      # list of clean attribute names

attr_index_map = {attr: idx for idx, attr in enumerate(attributes)}
# === CLASSIFIER ===
classifier = AttributeClassifier(num_classes=40).to(device)
CLASSIFIER_WEIGHTS_PATH = "./celeb_classifier/resnet_celeb_40_single.pth"
classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=device))
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Load SD
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    variant="fp16", torch_dtype=torch.float16, safety_checker=None
).to(device)
#pipe.load_lora_weights("./lora_weights/cifar10")
pipe.load_lora_weights("./lora_weights/celebsty-weight")

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
seed = 0
min_val = celeb_sd_params.MIN_VAL
max_val = celeb_sd_params.MAX_VAL
tier = "low"   # or "high"
if tier == "low":
    init_perturbation = celeb_sd_params.LOW_INIT
    perturbation_size = celeb_sd_params.LOW_SIZE
else:
    init_perturbation = celeb_sd_params.HIGH_INIT
    perturbation_size = celeb_sd_params.HIGH_SIZE 
best_left = 10
gen_num = 250
pop_size = 25
fitness_scores = []
all_img_lst = []
imgs_to_samp = 100
image_info = []
predicted_labels = []
height = 512
width = 512
num_inference_steps = 25
prompts = ["A photo of smiling-face person","A photo of curtain-bangs person", "A photo of eye-glasses person"]
for n in range(imgs_to_samp):
    seedSelect = seed+n
    generator = generator.manual_seed(seedSelect)
    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device).to(torch.float16)
    # Generate Prompt and expected label
    randprompt = random.choice(prompts)
   # best_prompt = None
   # best_score = -1
   # best_image = None
   # best_pred_label = None
    match = re.search(r"of\s([a-zA-Z-]+)", randprompt)
    if match:
        attr_key = match.group(1)  # e.g., smiling-face, curtain-bangs, eye-glasses
    else:
       attr_key = "unknown"

    # Map extracted keyword to numeric label index
    attr_label_map = {
         "smiling-face": 31,      # Smiling
         "curtain-bangs":5,     # Bangs
         "eye-glasses": 15        # Eyeglasses
    }

    expected_label = attr_label_map.get(attr_key, -1)  # Default to -1 if not found

    print(f"Chosen prompt: {randprompt}")
    print(f"Extracted attribute: {attr_key}")
    print(f"Expected label index: {expected_label}")


    with torch.inference_mode():
        init_img = pipe.tgate(prompt= randprompt, guidance_scale=3.5, num_inference_steps=num_inference_steps, latents=original_lv)["images"][0]
    tensor_image = transform(init_img).unsqueeze(0).to(device)
    logits = classifier(tensor_image)
    probs = torch.sigmoid(logits).squeeze() 
    # binary predictions (0/1) for ALL attributes
    #y_true = (probs >= 0.5).astype(np.int32)     # [num_attrs]

    # indices and mask of predicted-1 attributes
    # threshold at 0.5 → binary preds
    y_pred_bin = (probs > 0.5).to(torch.int32)       # torch int tensor

    # indices of active attributes
    ones_idx_t = (y_pred_bin == 1).nonzero(as_tuple=True)[0]   # torch indices
    ones_idx = ones_idx_t.cpu().tolist()
    ones_names = [attributes[i] for i in ones_idx]

    print("probs:", probs.detach().cpu().numpy())
    print("active indices:", ones_idx)
    print("active names:", ones_names)

    for i, p in enumerate(probs.detach().cpu().numpy()):
        print(f"{attributes[i]}: prob={p:.3f} → {int(y_pred_bin[i].item())}")

    attr_prob = probs[expected_label].item()
    original_label = int(attr_prob > 0.5)  # 1 = present, 0 = absent
    print(original_label)
    print("probability",probs)
    print("attribute_prob",attr_prob)

    if original_label == 1:
       filename = f"image_{saved_images}_X{expected_label}_Y{original_label}.png"
       init_img.save(os.path.join(proj_path, filename))
       # only log saved images to this CSV
       with open(csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([filename, expected_label, randprompt, original_label, attr_prob])
       print(f"Saved: {filename} | Prompt: {randprompt}")
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
                perturb_img = pipe.tgate([randprompt]*(pop_size),guidance_scale =1.4, generator= generator, 
                   num_inference_steps=num_inference_steps,
                   latents=indivs_lv,
                )["images"]
           # all_img_lst.append(perturb_img)
           torch.cuda.empty_cache()

           tensor_image2 =torch.stack([transform(image) for image in perturb_img])
           # tensor_image2 = transform(last_image)
           tensor_image2 = tensor_image2.to(device)
           all_logits = classifier(tensor_image2).detach()
          # perturb_label1 = np.argmax(all_logits).item()
           print(all_logits.shape)
           print("logits sample:", all_logits[0])       # first candidate’s logits
           print("logits min:", all_logits.min().item())
           print("logits max:", all_logits.max().item())
           print(tensor_image2.shape)
           os.makedirs(os.path.join(proj_path, 'generated_images'), exist_ok=True)
          
           #fitness_scores = [
           #    fitness_logit_thresh((all_logits[k_idx]).squeeze(), expected_label)
           #    for k_idx in range(pop_size)
           #]
           # Each candidate has 40 attribute logits. We select only the target attribute column.
           # Minimizing this logit will reduce the attribute probability (flip 1 → 0).
           fitness_scores = all_logits[:, expected_label].detach().cpu().numpy().tolist()

           print("print fitness",len(fitness_scores))
           print("fitness_scores", fitness_scores)
           # Find the minimum fitness score in the current generation
           current_min_index = np.argmin(fitness_scores)
           current_min_fitness = fitness_scores[current_min_index]
           print("cur",current_min_index)
           # Update best tracking variables if the current minimum is less than the tracked best
           if current_min_fitness < best_fitness_score:
              best_fitness_score = current_min_fitness
              best_image_tensor  = tensor_image2[current_min_index]  # Ensure a deep copy
              best_image_index = current_min_index

           # p_best = float(torch.sigmoid(all_logits[best_image_index]).view(-1)[expected_label])
           logits_best = all_logits[current_min_index].squeeze()
           p_best = float(torch.sigmoid(logits_best[expected_label]))
           # Perform selection
           selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True,
             )[-best_left:]
           print("selected-indices",selected_indices)
           for idx in selected_indices:
               print(f" idx={idx}, fitness={fitness_scores[idx]:.4f}")
           #selected_indices = np.argsort(fitness_scores)[:best_left]  # keep smallest (best) parents
           print(f"[stop] fit={current_min_fitness:.4f}, p_t={p_best:.3f}")
           now_best = np.min(fitness_scores)
           parent_pop = [now_pop[idx] for idx in selected_indices]
           print(parent_pop[-1].shape)
           print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
          # wandb.log({"ft_score":now_best})
           if now_best < 0:
              break
           elif now_best == prev_best:
               perturbation_size *= 0.75        #min(perturbation_size * 2, celeb_sd_params.LOW_SIZE * 1.5)
           else:
               perturbation_size = init_perturbation

           k_pop = []
           # print("Size of parent_pop:", len(parent_pop))
           # select k-idx for cross_over genes
           for k_idx in range(pop_size - best_left):
               mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
              # print("mom_idx:", mom_idx, "pop_idx:", pop_idx)
               spl_idx = np.random.choice(4, size=1)[0]
           # print("Size of parent_pop:", len(parent_pop))
           # select k-idx for cross_over genes
                         # print("mom_idx:", mom_idx, "pop_idx:", pop_idx)
               k_gene = torch.cat(
                   [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                   dim=1,
               )  # crossover

               # Mutation
               diffs = (k_gene != original_lv).float()       
               k_gene += (
                      (perturbation_size) * torch.randn(k_gene.size(), device=k_gene.device) * diffs
               )  # random adding noise only to diff places
               #k_gene = torch.clamp(k_gene, min_val, max_val)
               #interp_mask = binom_sampler.sample()
              # k_gene = interp_mask * original_lv + (1 - interp_mask) * k_gene
               k_pop.append(k_gene)
               # print("Size of k_pop:", len(k_pop))

           # Combine parent_pop and k_pop for the next generation
           now_pop = parent_pop + k_pop
           prev_best = now_best
           # Apply a final clamp across all now_pop before next iteration starts
           #now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

      
     # After the loop, save the last image if it exists
       if best_image_tensor is not None:
          tensor_image = best_image_tensor.to(device)
          tensor_image = tensor_image.unsqueeze(0).to(device)
          tensor_image_np= tensor_image.squeeze().detach().cpu().numpy()
          perturb_logit = classifier(tensor_image)
          perturb_probs = torch.sigmoid(perturb_logit).squeeze()
          attr_prob = perturb_probs[expected_label].item()
          # binary decision: 1 = present, 0 = absent
          perturb_label = int(attr_prob > 0.5)
          predicted_labels.append(perturb_label)
          all_img_lst.append(tensor_image_np)
          # Save the image as a numpy array
          print("Best image tensor shape:", best_image_tensor.shape)
          print("Tensor dtype before save:", tensor_image.dtype)
          print("Before normalization:")
          print("  Min:", best_image_tensor.min().item())
          print("  Max:", best_image_tensor.max().item())
          tensor_image1 = best_image_tensor.cpu().detach()
          # Normalize to [0, 1] range (visual saving)
          #min_val1 = tensor_image.min()
          #max_val1 = tensor_image.max()

          #if max_val1 > min_val1:
            # tensor_image = (tensor_image - min_val1) / (max_val1 - min_val1)
          #else:
            # tensor_image = torch.zeros_like(tensor_image)  # avoid divide-by-zero

          # Convert to uint8 [0, 255]
         # tensor_image = (tensor_image * 255).clamp(0, 255).to(torch.uint8)
          mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
          std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

          tensor_image = (tensor_image1 * std + mean).clamp(0,1)   # back to [0,1]
          img_uint8 = (tensor_image * 255).byte()

          #best_image_pil = transforms.ToPILImage()(img01)  # this expec
          best_image_pil = transforms.ToPILImage()(tensor_image)
          image_filename = f'image_{saved_images}_iteration{g_idx + 1}_X{expected_label}_Y{perturb_label}.png'
          last_image_path = os.path.join(proj_path, 'perturb-result', image_filename)
          best_image_pil.save(last_image_path)

          # Save the image as a numpy array
          np_filename = f'image_{saved_images}_iteration{g_idx + 1}_X{expected_label}_Y{perturb_label}'
          np.save(os.path.join(proj_path,'generated_images', np_filename), tensor_image_np)
       saved_images += 1
    else: 
       # IF no match, simply skip to the next iteration (no image is saved or processed)
       print(f"Label mismatch for image {saved_images}:expected {expected_label},got {original_label}")
# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(proj_path, "bound_imgpiz_sd.npy"), all_imgs)
