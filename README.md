# LLM-SLERP-Merge
Spherical Merge HuggingFace-Pytorch format Language Models for minimal feature loss of parent models.

# Spherical Linear Interpolation (SLERP) Model Merging

Traditionally, model merging often resorts to weight averaging which, although straightforward, might not always capture the intricate features of the models being merged. The SLERP technique in this script addresses this limitation, producing a blended model with characteristics smoothly interpolated from both parent models, ensuring the resultant model captures the essence of both its parents.

## Advantages over Standard Weight Averaging

1. **Smooth Transitions**: SLERP ensures smoother transitions between model parameters. This is especially significant when interpolating between high-dimensional vectors.
  
2. **Better Preservation of Characteristics**: Unlike weight averaging, which might dilute distinct features, SLERP preserves the curvature and characteristics of both models in high-dimensional spaces.

3. **Nuanced Blending**: SLERP takes into account the geometric and rotational properties of the models in the vector space, resulting in a blend that is more reflective of both parent models' characteristics.

## How to Use

1. Clone this repository.
```bash
git clone https://github.com/Digitous/LLM-SLERP-Merge.git
```
2. Navigate to the cloned directory.
```bash
cd LLM-SLERP-Merge
```
3. (Optional) Ensure you have the proper dependencies: numpy, torch, transformers, tkinter, and colorama; you can install them using:
```bash
pip install -r requirements.txt
```
4. Run the SLERP script.
```bash
python slerpmergelm.py
```
5. Follow the on-screen prompts to select the primary model, secondary model, and the directory to save the blended model. Ensure parent models are of the same architecture and parameter size (for example both LLaMa2 13B pretrained language models). The script will do the rest, spherical merging both parent models and saving the offspring model to the selected save directory. For added convenience, it will also scan both parent directories to see if one has a special_tokens_map.json and will proceed to copy all relevant tokenizer files from there to the child directory (in case both or neither contains the special_tokens_map, it will still copy necessary files to the child dir).

---

## License

This project is unlicensed and unrestricted on how far it's proliferated, updated, modified, maintined, integrated, and shared around. A kind reference to this repo as well as dvschultz's script (which inspired this work) at https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c would be nice.

---

**Contributors**: [Digitous](https://github.com/Digitous) & [CalderaAI](https://huggingface.co/CalderaAI) For retrofitting SLERP script for LLM (Pytorch+HF format) merging.

**Original Script** [dvschultz](https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c) For giving insights on how to go about Spherical Linear Interpolation with their script.

**Special Mention** [LostRuins](https://github.com/LostRuins) For first weight averaging script for LLMs (that we know of; without their work, none of this would have come to fruition).
