Convert my existing TerraVision satellite super-resolution project
for deployment on a Hugging Face ZeroGPU Gradio Space.

Do NOT rewrite or change the existing functionality/UI.

Requirements:

1. Keep the existing Sentinel-2 L2A → LDSR-S2/SEN2SR pipeline.
2. Keep 10-band 128×128 input.
3. Keep direct 4× super-resolution:
   128×128 → 512×512
4. Keep all existing analysis layers:
   RGB
   False Color
   SWIR
   NDVI
   NDWI
   NDBI
   NBR
   LDSR-S2 Uncertainty
5. Keep GeoTIFF generation and ZIP download.
6. Keep the current Gradio UI.
7. Keep the current uncertainty visualization/contrast improvement.
8. Make the GPU inference function compatible with Hugging Face ZeroGPU using:

   import spaces

   @spaces.GPU(duration=300)

9. Do not put model inference inside a normal CPU-only function.
10. Keep model loading outside the GPU inference function where appropriate.
11. Create/update:
    app.py
    requirements.txt
    README.md

12. requirements.txt must include the dependencies actually used by
    the existing project. Do not unnecessarily replace the project's
    existing PyTorch/CUDA installation.

13. Make sure the Gradio function returns exactly the same number of
    outputs as the UI components.

14. Make sure images returned to Gradio are sanitized and converted
    appropriately so ImageSlider does not produce:

    ValueError: Images of type float must be between -1 and 1

15. Make sure NaN/Inf values cannot break GeoTIFF generation.

16. Do not remove any existing functionality.

First inspect the entire existing project and identify the current
entry point, dependencies, model-loading code, GPU inference code,
and Gradio outputs.

Then make the minimum changes required for ZeroGPU deployment.