# LLM-SLERP-Merge
Spherical Merge HuggingFace-Pytorch format Language Models for minimal feature loss of parent models.

```
10 September 2023 - Important Update is on the way, I am sharing this note at the top here as it's as much a hotfix as it is
a feature. While I can't directly guarantee what was added last upload as a 'quality of life' feature that if it does its job
properly no one would notice it. Finetuned models with an extended vocabulary result in models that can't be merged with anything unless the merge script has logic to handle these events built in. I've seen the straightforward approach to truncate some of the busier model's tensors when loaded in memory right before merge ops and that's a brutal sacrifice. The last update was the first one to handle mismatched model sizes of the same pretrained family/B param size so the end user isn't suckerpunched by this very annoying issue, leaving them not much to go off of, wondering if a merge will or will not be possible with any given combination of models.

This is where mistakes were made. I pursued an additive approach that extends the smaller model in memory and when the parent models are merged, the vocab deposited in the child model's folder is double checked by the script and appended if needed.

But what's the issue? The extension method relied on adding tensors filled with zeroes. It made the merge script happy and the resulting model performs fine. We did have a basic function to check VS a deep Epsilon that told the operation to phone it in when it spotted tensors that were basically zeroes to the extend of what modern processors can handle. So the results were great, unless further merging from there occurs or the inference system does math a certain way or if the merge just sucked in the first place, making then math a lopsided catastrophe, resulting in a sideshow of shit haunted by NaN and inf values in your favorite LLM handler. All this for a convenience feature aiming to be completely invisible while preserving the most information.

The Hotfix: is in the lab, we're not backing down addressing disparate model shapes by extending the meek and the shy. We die on that hill, with shame if we have to. This time an elegant approach that extrapolates a meaningful representation of that model's data by taking the edge of two tensors and spacing them across as many tensors required to meet model parity - each extension tensor between them will be a gradient between their values. It's not perfect but we're not burning perfectly good information on a bigger model even if the hard way has to be done. Is this approach a little extra, and a reversion of a simple path? Well yes, and it's better.

I also owe it to llms for all the reckless brain surgery experiments I've done on them so there's that.
Enough writing an explainer book - I'll get the hotfix out soon. Send me hopes and dreams for support.

```

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
5. Follow the on-screen prompts to select the primary model, secondary model, and the directory to save the blended model. Ensure parent models are of the same architecture and parameter size (for example both LLaMa2 13B pretrained language models). The script will do the rest, spherical merging both parent models and saving the offspring model to the selected save directory. For added convenience, it will also scan both parent directories to see if one has a special_tokens_map.json and will proceed to copy all relevant tokenizer files from there to the child directory (in case both or neither contains the special_tokens_map, it will still copy necessary files to the child dir providing a model instantly ready to use when the process is complete).

---

## Convenience Feature

Some models, even of the same architecture and parameter size, may have a different vocab_size as defined in their config.json. For instance, LLaMa v1 and v2 13B have a standardized vocab of 32000 however, some pretrained LLaMa 13B models may deviate from this standard with a modified vocab of 32001 or 32032 and so on, which makes them incompatible for merging. We have added automatic model shape mismatch and vocab mismatch handling. User selected model pairs will be analyzed prior to merge. If there is a shape or vocab mismatch this script detects which model is short in vocab or shape, and an embedding is injected with empty tensors - compensating for disparities. The script will not continue to merge until it has ensured both models in memory are in complete shape and vocab parity. During the merge process, tensors beyond Epsilon = 1e-10 will be skipped, mitigating the potential of dividing by zero with the added tensors. When the merge is completed, the script ensures the vocab size in the config.json of the child model is correct. 

---

## License

This project is unlicensed and unrestricted on how far it's proliferated, updated, modified, maintined, integrated, and shared around. A kind reference to this repo as well as dvschultz's script (which inspired this work) at https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c would be nice.

---

**Contributors**: [Digitous](https://github.com/Digitous) & [CalderaAI](https://huggingface.co/CalderaAI) For retrofitting SLERP script for LLM (Pytorch+HF format) merging.

**Original Script** [dvschultz](https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c) For giving insights on how to go about Spherical Linear Interpolation with their script.

**Special Mention** [LostRuins](https://github.com/LostRuins) For first weight averaging script for LLMs (that we know of; without their work, none of this would have come to fruition).
