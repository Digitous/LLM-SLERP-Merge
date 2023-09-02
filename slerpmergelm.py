# LLM HF SLERP Merge

# Retrofitted from dvschultz's script at https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
# to work with Huggingface Pretrained Language Models [by Chasm (AKA Digitous) and CalderaAI (on HuggingFace)].
# Original language model linear interpolation methods pioneered by Concedo AKA LostRuins on Github and HF.

# Idea for SLERP on LLMs sparked by discussion in Automatic1111 Stable Diffusion UI feature request for SLERP
# model merging for image diffusion domain models.

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import subprocess
import os
import shutil
import json
import tkinter as tk
from tkinter import filedialog
from colorama import init, Fore, Style

newline = '\n'
def clear_console():
    if os.name == "nt":  # For Windows
        subprocess.call("cls", shell=True)
    else:  # For Linux and macOS
        subprocess.call("clear", shell=True)

clear_console()
print(f"{Fore.YELLOW}Starting {Fore.GREEN}spherical linear interpolation{Fore.YELLOW} script, please wait...{Style.RESET_ALL}")

def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation (batched)
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (torch.Tensor): Starting tensor
        v1 (torch.Tensor): Final tensor
    Returns:
        v2 (torch.Tensor): Interpolation tensor between v0 and v1
    '''

    # Convert tensors to a common format, float32
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)

    # Convert tensors to numpy arrays
    c = False
    if not isinstance(v0,np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1,np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles    
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    del v0_copy,v1_copy
    del v1

    if c:
        res = torch.from_numpy(v2) #.to("cuda")
    else:
        res = v2
    return res

def load_sharded_model(path):
    state_dict = {}
    shard_paths = [f for f in os.listdir(path) if f.startswith('pytorch_model') and f.endswith('.bin')]
    for shard_path in sorted(shard_paths, key=lambda x: int(x.split('-')[1])):
        shard = torch.load(os.path.join(path, shard_path), map_location='cpu')
        state_dict.update(shard)
    return {'state_dict': state_dict}

def select_path(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    return path

def load_model(path):
    if os.path.exists(os.path.join(path, 'pytorch_model.bin')):
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
        return {'state_dict': state_dict}
    else:
        return load_sharded_model(path)

def save_model(model, path):
    torch.save(model, path)

primary_model_path = select_path("Select the first model")
secondary_model_path = select_path("Select the second model")
blended_model_savedir = select_path("Select the directory to save the blended model")

primary_model = load_model(primary_model_path)
secondary_model = load_model(secondary_model_path)

v0 = primary_model['state_dict']
v1 = secondary_model['state_dict']

# Interpolating Parameters
for key in set(v0.keys()).union(set(v1.keys())):
    if key in v0 and key in v1:
        # Check if both values are tensors
        if isinstance(v0[key], torch.Tensor) and isinstance(v1[key], torch.Tensor):
            v0[key] = slerp((float(1.0) - 0.5), v0[key], v1[key])
        else:
            print(f"Skipping key {key} because it does not point to tensors.")
    if key in v1 and key not in v0:
        v0[key] = v1[key]
        del v1[key]

del secondary_model

print(f"{Fore.YELLOW}\nCopying necessary files and saving blended model to: {blended_model_savedir}{Style.RESET_ALL}")

for key, value in v0.items():
    if isinstance(value, np.ndarray):
        v0[key] = torch.tensor(value)

model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(primary_model_path))
model.load_state_dict(primary_model['state_dict'])
model.save_pretrained(blended_model_savedir, max_shard_size="20000MiB")

#save_model_path = blended_model_savedir + '/pytorch_model.bin'
#save_model(primary_model, save_model_path)

# List of files to copy to merged model dir
files_to_copy = ["special_tokens_map.json", "tokenizer_config.json", "vocab.json", "tokenizer.model", "generation_config.json", "added_tokens.json", "merges.txt"]
            
# Check for the existence of 'special_tokens_map.json' in both directories
first_model_has_special_tokens = os.path.exists(os.path.join(primary_model_path, "special_tokens_map.json"))
second_model_has_special_tokens = os.path.exists(os.path.join(secondary_model_path, "special_tokens_map.json"))
            
# Decide the source directory based on the presence of 'special_tokens_map.json'
if first_model_has_special_tokens and not second_model_has_special_tokens:
    src_dir = primary_model_path
elif second_model_has_special_tokens or not first_model_has_special_tokens:
    src_dir = secondary_model_path
            
# Copy each file to the new folder
for filename in files_to_copy:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(blended_model_savedir, filename)
    print(f"\nCopying files from dir: {src_path}")
    print(f"To dir: {dst_path}")
    try:
        shutil.copy2(src_path, dst_path)
    except FileNotFoundError:
        print("\nFile " + filename + " not found in " + src_dir + ". Skipping (likely not important).")  
