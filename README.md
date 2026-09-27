---
title: TerraVision
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
pinned: false
license: mit
---

# 🌍 TerraVision — Sentinel-2 Super Resolution & Analysis

**Sharper Earth. Brighter Decisions.**

TerraVision performs 4× satellite super-resolution on Sentinel-2 L2A imagery
using the ESA **LDSR-S2** latent diffusion model and **SEN2SR**, upscaling from
10 m to 2.5 m spatial resolution across all 10 bands.

---

## How to Use

1. **Click on the map** to select any location on Earth, or type coordinates manually.
2. Set a **date range** (a few days is enough; the pipeline picks the first available image).
3. Click **🚀 Generate Super-Resolution**.
4. Explore the analysis layers and download the georeferenced GeoTIFF ZIP.

---

## Output Layers

| Layer | Description |
|---|---|
| **RGB** | True-colour super-resolved composite |
| **False Color / NIR** | NIR–Red–Green (vegetation highlighting) |
| **SWIR Composite** | SWIR2–SWIR1–Red (geology & burn scars) |
| **NDVI** | Normalised Difference Vegetation Index |
| **NDWI** | Normalised Difference Water Index |
| **NDBI** | Normalised Difference Built-up Index |
| **NBR** | Normalised Burn Ratio |
| **LDSR-S2 Uncertainty** | Pixel-wise model uncertainty from the diffusion ensemble |

All layers are exported as **georeferenced GeoTIFF** files (EPSG as acquired)
and bundled into a single ZIP download.

---

## Models

- **LDSR-S2** — ESA / tacofoundation latent diffusion SR for RGB+NIR bands  
  Weights: [`tacofoundation/RS-SR-LTDF`](https://huggingface.co/tacofoundation/RS-SR-LTDF)
- **SEN2SR** — Swin Transformer SR for 20 m Sentinel-2 bands guided by LDSR-S2 output

---

## Technical Details

- Input: 10-band Sentinel-2 L2A at 128 × 128 pixels (10 m native resolution)
- Output: 10-band SR product at 512 × 512 pixels (2.5 m effective resolution)
- Uncertainty: 5-sample diffusion ensemble (`n_variations=5`, `sampling_steps=50`)
- Data source: [cubo](https://github.com/ESDS-Leipzig/cubo) → Microsoft Planetary Computer STAC

---

## Acknowledgements

Built on top of [OpenSR](https://github.com/ESA-PhiLab/OpenSR) and
[SEN2SR](https://github.com/ESA-PhiLab/SEN2SR) by ESA Phi-Lab / ESDS Leipzig.
