# LLM-SLERP-Merge
Spherical Merge Pytorch/HF format Language Models with minimal feature loss.

# Spherical Linear Interpolation (SLERP) Model Merging

Traditionally, model merging often resorts to weight averaging which, although straightforward, might not always capture the intricate features of the models being merged. The SLERP technique in this script addresses this limitation, producing a blended model with characteristics smoothly interpolated from both parent models, ensuring the resultant model captures the essence of both its parents.

## Advantages over Standard Weight Averaging

1. **Smooth Transitions**: SLERP ensures smoother transitions between model parameters. This is especially significant when interpolating between high-dimensional vectors.
  
2. **Better Preservation of Characteristics**: Unlike weight averaging, which might dilute distinct features, SLERP preserves the curvature and characteristics of both models in high-dimensional spaces.

3. **Nuanced Blending**: SLERP takes into account the geometric and rotational properties of the models in the vector space, resulting in a blend that is more reflective of both parent models' characteristics.

## How to Use

1. Clone this repository.
```bash
git clone https://github.com/yourusername/slerp-model-merging.git
```
2. Navigate to the cloned directory.
```bash
cd slerp-model-merging
```
3. Run the SLERP script.
```bash
python slerp_script.py
```
4. Follow the on-screen prompts to select the primary model, secondary model, and the directory to save the blended model.

---

## License

This project is unlicensed and I don't care how far it's proliferated, updated, modified, maintined, integrated, and shared around. A kind reference to this repo as well as dvschultz's script (which inspired this work) at https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c would be nice.

---

**Contributors**: [Digitous](https://github.com/Digitous) [CalderaAI](https://huggingface.co/CalderaAI)
**Original Script** [dvschultz](https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c)
