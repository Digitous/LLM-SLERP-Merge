# LLM-SLERP-Merge
Spherical Merge Pytorch/HF format Language Models with minimal feature loss.

# Spherical Linear Interpolation (SLERP) Model Merging

Merging or blending machine learning models is a technique used to combine the strengths and characteristics of two pre-trained models. The provided script in this repository achieves this by employing a method called Spherical Linear Interpolation (SLERP). Instead of simply averaging the parameters of the two models, SLERP offers a more nuanced and smooth interpolation, ensuring the resultant model captures the essence of both its parents.

## Description

Traditionally, model blending often resorts to weight averaging, which, although straightforward, might not always capture the intricate features of the models being merged. The SLERP technique in this script addresses this limitation, producing a blended model with characteristics smoothly interpolated from both parent models.

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

## Advantages over Standard Weight Averaging

1. **Smooth Transitions**: SLERP ensures smoother transitions between model parameters. This is especially significant when interpolating between high-dimensional vectors.
  
2. **Better Preservation of Characteristics**: Unlike weight averaging, which might dilute distinct features, SLERP preserves the curvature and characteristics of both models in high-dimensional spaces.

3. **Nuanced Blending**: SLERP takes into account the geometric and rotational properties of the models in the vector space, resulting in a blend that is more reflective of both parent models' characteristics.

## Conclusion

While standard weight averaging is a commonly used method, the SLERP technique offers a more sophisticated approach to model blending. For practitioners looking to harness the strengths of two distinct models without losing their unique features, this script is an invaluable tool.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Contributors**: [Your Name](https://github.com/yourusername)
