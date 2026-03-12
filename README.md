# 🎨 Son Goku LoRA Fine-Tuner — NoobAI-XL / Dual T4 Kaggle

Fine-tune [`Laxhar/noobai-XL-Vpred-1.0`](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0) (NoobAI-XL, Illustrious-XL fine-tune) on a Son Goku illustration dataset using LoRA on dual NVIDIA T4 GPUs (Kaggle free tier). Handles any image resolution, auto-captions missing labels, and pushes the adapter to HuggingFace Hub.

> **Why NoobAI-XL?** Best anatomical accuracy for complex anime subjects — dynamic poses, expressive faces, hands, and flowing hair — in 2026. Illustrious architecture with v-prediction scheduling delivers superior detail fidelity for character LoRAs.

---

## Quickstart

1. Upload Son Goku illustration images to a Kaggle dataset under `training_data/`
2. Add `HF_TOKEN` in **Add-ons → Secrets**
3. Set your `REPO_ID` in Cell 2 and Cell 6
4. Run all cells top to bottom

---

## Notebook Structure

| Cell | Purpose                                      |
| ---- | -------------------------------------------- |
| 1    | Install deps & verify HF auth                |
| 2    | Write training script to disk                |
| 3    | Copy dataset, validate & auto-caption        |
| 4    | Launch dual-T4 distributed training (~2–4 h) |
| 5    | Inference test — 2×2 image grid              |
| 6    | Push LoRA adapter + model card to Hub        |
| 7    | _(Optional)_ Merge LoRA into base & push     |
| 8    | Load from Hub & generate custom images       |

---

## Dataset Format

```
training_data/
├── img_001.png
├── img_001.txt   ← Danbooru-style tags (see below)
├── img_002.jpg
└── ...           ← .txt optional — auto-generated if missing
```

- **Supported formats:** `.png` `.jpg` `.jpeg` `.webp`
- **Any resolution accepted** — padded to square, then resized to 1024×1024 (no cropping)
- **Recommended count:** 150–300 high-quality, diverse illustrations
- **Tag style:** Danbooru (NoobAI's native language)
- **Trigger token:** `fcs` — always include at inference time

### Caption format

```
fcs, son_goku, orange_gi, spiky_hair, dynamic_pose,
cel_shading, high_contrast_shadows, masterpiece, best_quality, absurdres
```

**Always end captions with** `masterpiece, best_quality, absurdres`.

### Recommended Goku-specific tags

| Category       | Tags                                                            |
| -------------- | --------------------------------------------------------------- |
| Identity       | `son_goku`, `spiky_hair`, `black_hair`, `wristbands`            |
| Outfits        | `orange_gi`, `blue_undershirt`, `gi`, `turtle_hermit_uniform`   |
| Poses          | `dynamic_pose`, `fighting_stance`, `flying`, `powering_up`      |
| Power effects  | `aura`, `energy_blast`, `kamehameha`, `super_saiyan`, `glowing` |
| Expressions    | `determined_expression`, `shouting`, `smiling`, `serious`       |
| Composition    | `full_body`, `portrait`, `action_scene`, `dramatic_angle`       |
| Style boosters | `cel_shading`, `sharp_linework`, `high_contrast_shadows`        |

---

## Key Config (Cell 2)

| Param        | Default | Notes                                                  |
| ------------ | ------- | ------------------------------------------------------ |
| `TRIGGER`    | `fcs`   | Include in every prompt at inference                   |
| `LORA_RANK`  | `32`    | Increase to `64` for stronger style capture            |
| `LORA_ALPHA` | `64`    | Keep at `rank × 2`                                     |
| `EPOCHS`     | `3`     | 4–5 for smaller datasets (<100 images)                 |
| `LR`         | `1e-4`  | Lower to `5e-5` for fine detail refinement             |
| `RESOLUTION` | `1024`  | NoobAI-XL native resolution                            |
| `BATCH_SIZE` | `1`     | T4 memory constraint — do not increase                 |
| `GRAD_ACCUM` | `4`     | Effective batch = `BATCH_SIZE × GRAD_ACCUM × num_GPUs` |
| `SEED`       | `42`    | Fixed for reproducible training                        |

---

## ⚠️ V-Prediction — Critical Note

NoobAI-XL is a **v-prediction** model. You **must** configure the scheduler correctly at inference — the wrong scheduler produces grey, washed-out, or pure noise output.

```python
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
)
```

---

## Inference — Quick Start

```python
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from peft import PeftModel
import torch

# Load base model across both GPUs
pipe = StableDiffusionXLPipeline.from_pretrained(
    "Laxhar/noobai-XL-Vpred-1.0",
    torch_dtype=torch.float16,
    device_map="balanced",
    max_memory={0: "14GiB", 1: "14GiB"},
)

# Required: v-prediction scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
)

# Inject LoRA adapter
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "YOUR_HF_USERNAME/noobai-xl-lora",
    is_trainable=False,
)
pipe.unet.eval()

# Memory optimizations for T4
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# Generate
image = pipe(
    "fcs, son_goku, orange_gi, spiky_hair, super_saiyan, aura, "
    "dynamic_pose, dramatic_lighting, masterpiece, best_quality, absurdres",
    negative_prompt="lowres, bad_anatomy, bad_hands, missing_fingers, "
                    "worst_quality, low_quality, jpeg_artifacts, blurry, "
                    "3d, photo, realistic",
    num_inference_steps=28,
    guidance_scale=7.0,
    width=1024,
    height=1024,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
```

### Test prompts used in Cell 5 (2×2 grid)

| #   | Prompt focus       | Key tags                                                       |
| --- | ------------------ | -------------------------------------------------------------- |
| 1   | Style reference    | `son_goku, sitting, wooden_chair, cel_shading, sharp_linework` |
| 2   | Portrait           | `son_goku, portrait, detailed_eyes, soft_lighting`             |
| 3   | Action / full body | `son_goku, dynamic_pose, full_body, wind, dramatic_lighting`   |
| 4   | Action pose        | `son_goku, sitting, crossed_legs, elegant_interior`            |

---

## Loading Modes (Cell 8)

Cell 8 supports two loading strategies, switchable via `LOAD_MODE`:

| Mode        | `LOAD_MODE` | Description                                               |
| ----------- | ----------- | --------------------------------------------------------- |
| **Adapter** | `"adapter"` | Loads base model + lightweight LoRA adapter (recommended) |
| **Merged**  | `"merged"`  | Loads a fully fused standalone model (no base required)   |

The merged model (Cell 7) fuses LoRA weights into the base UNet — useful for deployment but pushes a full ~6GB model to Hub instead of a small adapter.

---

## Training Details

| Setting           | Value                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| Base model        | `Laxhar/noobai-XL-Vpred-1.0` (Illustrious-XL)                                                                      |
| Scheduler         | v-prediction + zero terminal SNR                                                                                   |
| LoRA rank / alpha | 32 / 64                                                                                                            |
| Target modules    | `to_q`, `to_k`, `to_v`, `to_out.0`, `proj_in`, `proj_out`, `ff.net.0.proj`, `ff.net.2`, `add_k_proj`, `add_v_proj` |
| Resolution        | 1024×1024 (padded to square, no cropping)                                                                          |
| Epochs            | 3                                                                                                                  |
| Optimizer         | AdamW (`lr=1e-4`, `weight_decay=1e-2`)                                                                             |
| LR schedule       | Cosine annealing (`eta_min=1e-6`)                                                                                  |
| Mixed precision   | fp16                                                                                                               |
| Hardware          | 2× NVIDIA T4 16 GB (Kaggle free tier)                                                                              |
| Multi-GPU         | `accelerate` + `--num_processes=2`                                                                                 |
| Trigger token     | `fcs`                                                                                                              |
| Tag style         | Danbooru                                                                                                           |

---

## Troubleshooting

| Symptom                         | Cause & Fix                                                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Grey / washed-out output        | Wrong scheduler — verify `prediction_type="v_prediction"`, `rescale_betas_zero_snr=True`, `timestep_spacing="trailing"` |
| Noisy / incoherent output       | Same as above, or `guidance_scale` too low — try 6.0–8.0                                                                |
| Weak style / trigger not firing | Increase `LORA_RANK=64`, `EPOCHS=4–5`, or lower `LR=5e-5`                                                               |
| OOM on T4                       | Keep `RESOLUTION=768` for smoke test, increase to 1024 after                                                            |
| Bad anatomy / distorted poses   | Add more varied pose images to dataset; aim for 150+ images                                                             |
| Auto-captions too generic       | Write manual `.txt` files with specific Danbooru tags                                                                   |

---

## File Structure

```
lora_output/
├── adapter_config.json     ← PEFT adapter metadata
├── adapter_model.safetensors
├── tokenizer/
├── tokenizer_2/
├── scheduler/
└── README.md               ← HuggingFace model card (written by Cell 6)
```

<!-- # 🎨 Anime LoRA Fine-Tuner — NoobAI-XL / Dual T4 Kaggle

Fine-tune `Laxhar/noobai-XL-Vpred-1.0` (NoobAI-XL, Illustrious-XL fine-tune) on your own illustration dataset using LoRA on dual NVIDIA T4 GPUs (Kaggle free tier). Handles any image resolution, auto-captions missing labels, and pushes the adapter to HuggingFace Hub.

> **Why NoobAI-XL?** Best anatomical accuracy for complex subjects — feet, toes, hands — in 2026. Illustrious architecture with v-prediction scheduling for superior detail fidelity.

## Quickstart

1. Upload images to a Kaggle dataset under `training_data/`
2. Add `HF_TOKEN` in **Add-ons → Secrets**
3. Set your `REPO_ID` in Cell 2 and Cell 6
4. Run all cells top to bottom

## Dataset Format

```
training_data/
├── img_001.png
├── img_001.txt   ← Danbooru-style tags (see below)
├── img_002.jpg
└── ...           ← .txt optional — auto-generated if missing
```

- **Supported formats:** `.png` `.jpg` `.jpeg` `.webp`
- **Any resolution accepted** — padded to square, then resized to 1024×1024
- **Recommended count:** 150–300 high-quality, diverse images
- **Tag style:** Danbooru (NoobAI's native language)
- **Trigger token:** `fcs` — always include at inference time

### Caption example

```
fcs, 1girl, blue_qipao, rope_bondage, bare_feet, detailed_toes,
cel_shading, high_contrast_shadows, masterpiece, best_quality, absurdres
```

Always end captions with `masterpiece, best_quality, absurdres`.

## ⚠️ V-Prediction — Critical Note

NoobAI-XL is a **v-prediction** model. At inference you **must** use:

```python
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
)
```

Wrong scheduler → grey, washed-out, or noisy output.

## Notebook Structure

| Cell | Purpose                               |
| ---- | ------------------------------------- |
| 1    | Install deps & verify HF auth         |
| 2    | Write training script to disk         |
| 3    | Copy dataset, validate & auto-caption |
| 4    | Launch dual-T4 distributed training   |
| 5    | Inference test — 2×2 image grid       |
| 6    | Push LoRA adapter + model card to Hub |

## Key Config (Cell 2)

| Param        | Default | Notes                                     |
| ------------ | ------- | ----------------------------------------- |
| `LORA_RANK`  | 32      | Increase to 64 for stronger style capture |
| `EPOCHS`     | 3       | 4–5 for smaller datasets (<100 images)    |
| `LR`         | 1e-4    | Lower to 5e-5 for fine detail refinement  |
| `RESOLUTION` | 1024    | NoobAI-XL native                          |
| `TRIGGER`    | `fcs`   | Include in every prompt at inference      |

## Inference

```python
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "Laxhar/noobai-XL-Vpred-1.0", torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
)

pipe.unet.load_adapter("YOUR_HF_USERNAME/your-lora-repo")

image = pipe(
    "fcs, 1girl, detailed_shading, masterpiece, best_quality, absurdres",
    negative_prompt="lowres, bad_anatomy, worst_quality, bad_feet",
    num_inference_steps=28, guidance_scale=7.0,
    width=1024, height=1024,
).images[0]
``` -->
