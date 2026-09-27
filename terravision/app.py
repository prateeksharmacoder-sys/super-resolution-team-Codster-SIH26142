# -*- coding: utf-8 -*-
"""
TerraVision — Sentinel-2 Super Resolution & Analysis
Hugging Face ZeroGPU Gradio Space
"""

# ============================================================
# IMPORT spaces FIRST — must precede torch / any CUDA import
# ============================================================
import spaces

# ============================================================
# STANDARD IMPORTS
# ============================================================
import os
import tempfile
import zipfile

import mlstac
import torch
import cubo
import numpy as np
import rasterio
import sen2sr
import requests
from datetime import datetime
import gradio as gr
from PIL import Image

# ============================================================
# ZEROGPU DURATION CONFIGURATION
# Increase this value if the Space account / hardware allows.
# Free-tier ZeroGPU can reject requests with very large durations.
# ============================================================
GPU_DURATION = 120  # seconds

# ============================================================
# DEVICE
# ZeroGPU handles CUDA emulation at module level.
# ============================================================
device = torch.device("cuda")

# ============================================================
# MODEL DOWNLOAD & LOADING
# Must be at module scope, outside @spaces.GPU.
# ZeroGPU handles weight migration automatically.
# ============================================================
if not os.path.isfile("model/LDSRS2-SEN2SR/mlm.json"):
    mlstac.download(
        file="https://huggingface.co/tacofoundation/RS-SR-LTDF/resolve/main/main/mlm.json",
        output_dir="model/LDSRS2-SEN2SR/"
    )

model = mlstac.load("model/LDSRS2-SEN2SR/").compiled_model(device=device)
model = model.to(device)

# ============================================================
# HELPER: SAVE TENSOR AS GEOTIFF
# ============================================================

def save_tensor_as_geotiff(tensor, attrs, out_path, super_resolved=False, sr_factor=4):
    """
    Save a PyTorch tensor as a georeferenced GeoTIFF using metadata in attrs.

    Parameters:
        tensor (torch.Tensor or np.ndarray): shape (bands, H, W), values in 0-1.
        attrs (dict): Metadata from LR image (.attrs).
        out_path (str): Output file path (.tif).
        super_resolved (bool): If True, assumes image is SR upscaled by sr_factor.
        sr_factor (int): SR upscale factor.
    """
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()

    # Guard against NaN / Inf before uint16 conversion
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    # Scale and clip
    arr = (tensor * 10000).clip(0, 10000).astype(np.uint16)

    # Original georef info
    pixel_size = attrs["resolution"]
    edge_size   = attrs["edge_size"]
    central_x   = attrs["central_x"]
    central_y   = attrs["central_y"]
    epsg        = attrs["epsg"]

    # Bounding box remains the same
    total_extent = edge_size * pixel_size
    half_extent  = total_extent / 2
    ul_x = central_x - half_extent
    ul_y = central_y + half_extent

    # If SR, update pixel size only (dimensions are already upsampled)
    if super_resolved:
        pixel_size = pixel_size / sr_factor

    # Define geotransform
    transform = rasterio.transform.from_origin(ul_x, ul_y, pixel_size, pixel_size)

    # Save
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
    ) as dst:
        dst.write(arr)


# ============================================================
# MAP CONFIGURATION
# ============================================================

DEFAULT_LAT = 39.39785676571274
DEFAULT_LON = -0.3798517619438821

# LEAFLET_HEAD removed — Gradio 5 gr.HTML does not support 'head'.
# Leaflet CSS/JS are inlined into MAP_HTML_VALUE below.


# ============================================================
# MAP JAVASCRIPT
# ============================================================

MAP_JS = """
// ============================================================
// TERRAVISION LEAFLET MAP
// ============================================================

const mapDiv = document.getElementById("terrativision-map");

if (!mapDiv) {

    console.error("❌ TerraVision map container not found.");

} else if (mapDiv._terraVisionMapInitialized) {

    console.log("ℹ️ TerraVision map already initialized.");

} else {

    mapDiv._terraVisionMapInitialized = true;

    const defaultLat = __DEFAULT_LAT__;
    const defaultLon = __DEFAULT_LON__;

    // --------------------------------------------------------
    // CREATE MAP
    // --------------------------------------------------------

    const map = L.map(mapDiv).setView(
        [defaultLat, defaultLon],
        10
    );

    // --------------------------------------------------------
    // STREET MAP
    // --------------------------------------------------------

    const osmLayer = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            maxZoom: 19,
            attribution: "&copy; OpenStreetMap contributors"
        }
    ).addTo(map);

    // --------------------------------------------------------
    // SATELLITE MAP
    // --------------------------------------------------------

    const satelliteLayer = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
            maxZoom: 19,
            attribution: "Tiles &copy; Esri"
        }
    );

    // --------------------------------------------------------
    // LAYER CONTROL
    // --------------------------------------------------------

    L.control.layers(
        {
            "Street Map": osmLayer,
            "Satellite": satelliteLayer
        }
    ).addTo(map);

    // --------------------------------------------------------
    // MARKER
    // --------------------------------------------------------

    let marker = L.marker(
        [defaultLat, defaultLon]
    ).addTo(map);

    // --------------------------------------------------------
    // LOCATION DISPLAY
    // --------------------------------------------------------

    const locationDisplay =
        document.querySelector("#map-location-display");

    function updateLocationDisplay(lat, lon) {

        if (locationDisplay) {

            locationDisplay.innerHTML =
                "📍 <b>Selected:</b> " +
                lat.toFixed(8) +
                ", " +
                lon.toFixed(8);
        }
    }

    updateLocationDisplay(
        defaultLat,
        defaultLon
    );

    marker.bindPopup(
        "<b>Selected Location</b><br>" +
        defaultLat.toFixed(8) +
        ", " +
        defaultLon.toFixed(8)
    );

    // --------------------------------------------------------
    // MAP CLICK
    // --------------------------------------------------------

    map.on("click", function(e) {

        const lat = e.latlng.lat;
        const lon = e.latlng.lng;

        console.log(
            "📍 TerraVision selected:",
            lat,
            lon
        );

        // Move marker
        marker.setLatLng([lat, lon]);

        // Update popup
        marker.bindPopup(
            "<b>Selected Location</b><br>" +
            lat.toFixed(8) +
            ", " +
            lon.toFixed(8)
        ).openPopup();

        // Update visible selected text
        updateLocationDisplay(lat, lon);

// --------------------------------------------------------
// UPDATE GRADIO LATITUDE / LONGITUDE
// --------------------------------------------------------

function updateGradioNumber(elemId, value) {

    const container = document.getElementById(elemId);

    if (!container) {
        console.error(
            "❌ Gradio component not found:",
            elemId
        );
        return;
    }

    const input = container.querySelector("input");

    if (!input) {
        console.error(
            "❌ Input element not found:",
            elemId
        );
        return;
    }

    // Use the native HTML value setter.
    // This is important because Gradio uses a controlled input.
    const nativeSetter =
        Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype,
            "value"
        ).set;

    nativeSetter.call(
        input,
        String(value)
    );

    // Tell Gradio that the user changed the input
    input.dispatchEvent(
        new Event("input", {
            bubbles: true
        })
    );

    input.dispatchEvent(
        new Event("change", {
            bubbles: true
        })
    );

    console.log(
        "✅ Updated Gradio:",
        elemId,
        value
    );
}

updateGradioNumber(
    "latitude_input",
    lat
);

updateGradioNumber(
    "longitude_input",
    lon
);

    });

    // --------------------------------------------------------
    // FIX MAP SIZE
    // --------------------------------------------------------

    setTimeout(function() {

        map.invalidateSize();

    }, 500);

    // Store map reference
    element._terraVisionMap = map;

    console.log(
        "✅ TerraVision Leaflet map initialized successfully."
    );
}
"""

# Insert Python values without using an f-string
MAP_JS = MAP_JS.replace(
    "__DEFAULT_LAT__",
    str(DEFAULT_LAT)
).replace(
    "__DEFAULT_LON__",
    str(DEFAULT_LON)
)

# ============================================================
# COMPLETE MAP HTML
# Gradio 5 gr.HTML has no 'head' or 'js_on_load' parameters.
# Inline Leaflet CSS, Leaflet JS, the map div, and the init
# script all inside the value string.
# ============================================================
MAP_HTML_VALUE = f"""
<link rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="terrativision-map"
     style="width:100%;height:400px;border-radius:14px;
            overflow:hidden;border:1px solid rgba(128,128,128,0.35);"></div>
<script>
{MAP_JS}
</script>
"""


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

#comparison_slider {
    width: 100% !important;
    max-width: 560px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

#comparison_slider img {
    object-fit: contain !important;
}

.analysis-image {
    width: 100% !important;
}


/* ----------------------------------------------------------
   MAP
---------------------------------------------------------- */

#terrativision-map {
    width: 100%;
    height: 400px;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(128,128,128,0.35);
}

#map-location-display {
    margin-top: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(128,128,128,0.10);
    font-size: 13px;
    text-align: center;
}
"""

# ============================================================
# IMAGE SANITIZATION HELPERS
# ============================================================

def clean_rgb(image):
    image = np.asarray(image, dtype=np.float32)

    # Remove NaN / Inf
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Keep normalized RGB range
    image = np.clip(image, 0.0, 1.0)

    # Convert to uint8 for Gradio
    return (image * 255.0).round().astype(np.uint8)


def clean_index(image):
    image = np.asarray(image, dtype=np.float32)

    # Remove NaN / Inf
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )

    # NDVI / NDWI / NDBI / NBR range
    image = np.clip(image, -1.0, 1.0)

    # Convert [-1, 1] -> [0, 255]
    image = (image + 1.0) / 2.0

    return (image * 255.0).round().astype(np.uint8)


def clean_uncertainty(image):
    image = np.asarray(image, dtype=np.float32)

    # Remove invalid values
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Find useful range of actual uncertainty values
    valid = image[np.isfinite(image)]

    if valid.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    # Contrast stretching using percentiles
    low  = np.percentile(valid, 2)
    high = np.percentile(valid, 98)

    # Prevent divide-by-zero
    if high <= low:
        high = low + 1e-6

    # Stretch actual uncertainty range to 0-1
    image = (image - low) / (high - low)

    image = np.clip(image, 0.0, 1.0)

    # Convert to display image
    return (image * 255.0).round().astype(np.uint8)


# ============================================================
# MAIN PROCESSING PIPELINE
# Called from inside @spaces.GPU so all GPU ops run there.
# ============================================================

def generate_satellite_products(
    latitude,
    longitude,
    start_date,
    end_date
):
    """
    Complete TerraVision processing pipeline:

    Location + dates
        |
    Sentinel-2 L2A
        |
    ESA LDSR-S2 + SEN2SR
        |
    10-band 2.5m SR
        |
    Analysis layers
        |
    Preview images + GeoTIFFs
    """

    print("========================================")
    print("Starting satellite processing")
    print("========================================")

    # --------------------------------------------------
    # 1. Get Sentinel-2 L2A data
    # --------------------------------------------------

    print("\n[1/6] Fetching Sentinel-2 L2A...")

    edge_size = 128

    da_new = cubo.create(
        lat=float(latitude),
        lon=float(longitude),
        collection="sentinel-2-l2a",
        bands=[
            "B02", "B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B11", "B12"
        ],
        start_date=str(start_date),
        end_date=str(end_date),
        edge_size=edge_size,
        resolution=10
    )

    if da_new.sizes.get("time", 0) == 0:
        raise ValueError(
            "No Sentinel-2 image was found for the selected "
            "location and date range."
        )

    print("Available images:", da_new.sizes.get("time", 0))

    # For the first prototype, use the first available image.
    image_index = 0

    # --------------------------------------------------
    # 2. Prepare 10-band input
    # --------------------------------------------------

    print("\n[2/6] Preparing 10-band input...")

    original_numpy = (
        da_new[image_index]
        .compute()
        .to_numpy()
    ).astype("float32")

    low_res = torch.from_numpy(original_numpy).float()

    # Sentinel-2 reflectance scaling
    low_res = low_res / 10_000

    # Original Sentinel-2 RGB (10 m)
    original_rgb = np.stack([
        original_numpy[2],  # B04 - Red
        original_numpy[1],  # B03 - Green
        original_numpy[0],  # B02 - Blue
    ], axis=-1)

    low_res = low_res.to(device)

    print("Input:", tuple(low_res.shape))

    # --------------------------------------------------
    # 3. ESA LDSR-S2 + SEN2SR  [GPU]
    # --------------------------------------------------

    print("\n[3/6] Running ESA super-resolution...")

    with torch.inference_mode():

        sr_tensor = sen2sr.predict_large(
            model=model,
            X=low_res,
            overlap=16
        )

    print("SR output:", tuple(sr_tensor.shape))

    # --------------------------------------------------
    # 4. Extract bands
    # --------------------------------------------------

    sr = sr_tensor.detach().cpu().numpy()

    B02 = sr[0]
    B03 = sr[1]
    B04 = sr[2]
    B05 = sr[3]
    B06 = sr[4]
    B07 = sr[5]
    B08 = sr[6]
    B8A = sr[7]
    B11 = sr[8]
    B12 = sr[9]

    # --------------------------------------------------
    # 5. Generate analysis layers
    # --------------------------------------------------

    print("\n[4/6] Generating analysis layers...")

    eps = 1e-8

    rgb = np.stack(
        [B04, B03, B02],
        axis=-1
    )

    false_color = np.stack(
        [B08, B04, B03],
        axis=-1
    )

    swir = np.stack(
        [B12, B11, B04],
        axis=-1
    )

    ndvi = (B08 - B04) / (B08 + B04 + eps)

    ndwi = (B03 - B08) / (B03 + B08 + eps)

    ndbi = (B11 - B08) / (B11 + B08 + eps)

    nbr  = (B08 - B12) / (B08 + B12 + eps)

    # --------------------------------------------------
    # 6. Uncertainty  [GPU]
    # --------------------------------------------------

    print("\n[5/6] Calculating LDSR-S2 uncertainty...")

    deep_model = model.sr_model.sr_model

    lr_4band = torch.stack([
        low_res[0],  # B02
        low_res[1],  # B03
        low_res[2],  # B04
        low_res[6],  # B08
    ], dim=0).unsqueeze(0)

    with torch.inference_mode():

        uncertainty_tensor = deep_model.uncertainty_map(
            lr_4band,
            n_variations=5,
            sampling_steps=50
        )

    uncertainty_np = (
        uncertainty_tensor
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    print("Uncertainty:", uncertainty_np.shape)

    # --------------------------------------------------
    # 7. Save GeoTIFFs
    # --------------------------------------------------

    print("\n[6/6] Creating GeoTIFF outputs...")

    output_dir = tempfile.mkdtemp(
        prefix="terravision_"
    )

    def save_layer(array, filename):
        path = os.path.join(
            output_dir,
            filename
        )

        # RGB/composite
        if array.ndim == 3:

            tensor = torch.from_numpy(
                np.transpose(array, (2, 0, 1))
            ).float()

        # Single-band
        else:

            tensor = torch.from_numpy(
                array
            ).float().unsqueeze(0)

        save_tensor_as_geotiff(
            tensor,
            da_new[image_index].attrs,
            out_path=path,
            super_resolved=True
        )

        return path

    paths = {}

    paths["SR RGB"] = save_layer(
        rgb,
        "sr_rgb.tif"
    )

    paths["False Color"] = save_layer(
        false_color,
        "false_color.tif"
    )

    paths["SWIR"] = save_layer(
        swir,
        "swir.tif"
    )

    paths["NDVI"] = save_layer(
        ndvi,
        "ndvi.tif"
    )

    paths["NDWI"] = save_layer(
        ndwi,
        "ndwi.tif"
    )

    paths["NDBI"] = save_layer(
        ndbi,
        "ndbi.tif"
    )

    paths["NBR"] = save_layer(
        nbr,
        "nbr.tif"
    )

    paths["Uncertainty"] = save_layer(
        uncertainty_np,
        "uncertainty.tif"
    )

    # --------------------------------------------------
    # ZIP all GeoTIFFs
    # --------------------------------------------------

    zip_path = os.path.join(
        output_dir,
        "terravision_layers.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for name, path in paths.items():

            z.write(
                path,
                arcname=os.path.basename(path)
            )

    print("\n========================================")
    print("PROCESSING COMPLETE")
    print("========================================")

    return {
        "rgb": rgb,
        "original_rgb": original_rgb,
        "false_color": false_color,
        "swir": swir,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndbi": ndbi,
        "nbr": nbr,
        "uncertainty": uncertainty_np,
        "files": paths,
        "zip": zip_path
    }


# ============================================================
# GRADIO HANDLER — GPU-DECORATED
# All GPU computation (sen2sr.predict_large, uncertainty_map)
# runs inside generate_satellite_products which is called here.
# ============================================================

@spaces.GPU(duration=GPU_DURATION)
def run_app(latitude, longitude, start_date, end_date):

    results = generate_satellite_products(
        latitude,
        longitude,
        start_date,
        end_date
    )

    # -------------------------------------------------
    # ORIGINAL RGB
    # Sentinel-2 reflectance: 0-10000 -> 0-1
    # -------------------------------------------------
    original_rgb = np.asarray(
        results["original_rgb"],
        dtype=np.float32
    ) / 10000.0

    original_rgb = clean_rgb(original_rgb)

    # -------------------------------------------------
    # SUPER-RESOLVED RGB
    # -------------------------------------------------
    sr_rgb = clean_rgb(results["rgb"])

    # -------------------------------------------------
    # IMAGE SLIDER
    # -------------------------------------------------
    comparison_images = (
        original_rgb,
        sr_rgb
    )

    # -------------------------------------------------
    # ANALYSIS LAYERS
    # -------------------------------------------------
    false_color_img = clean_rgb(results["false_color"])

    swir_img = clean_rgb(results["swir"])

    ndvi_img = clean_index(results["ndvi"])

    ndwi_img = clean_index(results["ndwi"])

    ndbi_img = clean_index(results["ndbi"])

    nbr_img = clean_index(results["nbr"])

    uncertainty_img = clean_uncertainty(
        results["uncertainty"]
    )

    # -------------------------------------------------
    # ZIP FILE
    # -------------------------------------------------
    zip_file_path = results["zip"]

    # -------------------------------------------------
    # RETURN EXACTLY 9 OUTPUTS
    # -------------------------------------------------
    return (
        comparison_images,
        false_color_img,
        swir_img,
        ndvi_img,
        ndwi_img,
        ndbi_img,
        nbr_img,
        uncertainty_img,
        zip_file_path
    )


# ============================================================
# TERRAVISION APP
# ============================================================

with gr.Blocks(
    title="TerraVision — Sentinel-2 Super Resolution",
    css=custom_css
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        """
        # 🌍 TerraVision
        ### Sentinel-2 Super Resolution & Analysis

        **Sharper Earth. Brighter Decisions.**
        """
    )


    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    with gr.Row():

        # ====================================================
        # LEFT SIDEBAR
        # ====================================================

        with gr.Column(scale=1):

            gr.Markdown("### 📍 Select Location")


            # ------------------------------------------------
            # INTERACTIVE MAP
            # ------------------------------------------------

            map_html = gr.HTML(
                # Leaflet CSS/JS + map div + init script all inlined.
                # Gradio 5 gr.HTML does not support 'head' or 'js_on_load'.
                value=MAP_HTML_VALUE,
                elem_id="map-container-wrapper"
            )



            # Separate component for the location display
            map_location_output = gr.HTML(
                value=f"""
                <div id="map-location-display">
                    📍 <b>Selected:</b>
                    {DEFAULT_LAT:.8f}, {DEFAULT_LON:.8f}
                </div>
                """
            )


            gr.Markdown(
                "💡 **Click anywhere on the map to select a location.**"
            )


            # ------------------------------------------------
            # COORDINATES
            # ------------------------------------------------

            gr.Markdown("### 📌 Coordinates")

            latitude = gr.Number(
                label="Latitude",
                value=DEFAULT_LAT,
                precision=8,
                elem_id="latitude_input"
            )

            longitude = gr.Number(
                label="Longitude",
                value=DEFAULT_LON,
                precision=8,
                elem_id="longitude_input"
            )
             # ============================================================
# MAP -> GRADIO COORDINATE UPDATE
# ============================================================

            map_html.click(
                fn=None,
                inputs=[],
                outputs=[latitude, longitude],
                js="""
                () => {
                    const coords =
                        window.terraVisionSelectedCoordinates;

                    if (!coords) {
                        return [null, null];
                    }

                    return [
                        coords.latitude,
                        coords.longitude
                    ];
                }
                """
            )



            # ------------------------------------------------
            # DATE RANGE
            # ------------------------------------------------

            gr.Markdown("### 📅 Date Range")


            start_date = gr.Textbox(
                label="Start Date",
                value="2024-10-29"
            )


            end_date = gr.Textbox(
                label="End Date",
                value="2024-11-01"
            )



            # ------------------------------------------------
            # GENERATE BUTTON
            # ------------------------------------------------

            generate_button = gr.Button(
                "🚀 Generate Super-Resolution",
                variant="primary",
                size="lg"
            )


            # ------------------------------------------------
            # MODEL INFORMATION
            # ------------------------------------------------

            gr.Markdown(
                """
                **Model:** ESA LDSR-S2 + SEN2SR

                **Input:** Sentinel-2 L2A

                **Output:** 2.5 m

                **Bands:** 10
                """
            )


        # ====================================================
        # RIGHT CONTENT
        # ====================================================

        with gr.Column(scale=3):

            gr.Markdown(
                "## Results & Analysis Layers"
            )


            # ------------------------------------------------
            # MAIN BEFORE / AFTER
            # ------------------------------------------------

            comparison_output = gr.ImageSlider(

                label="10 m -> 2.5 m Super-Resolution",

                type="numpy",

                image_mode="RGB",

                height=500,
            )


            # ------------------------------------------------
            # FALSE COLOR + SWIR
            # ------------------------------------------------

            with gr.Row():

                false_color_output = gr.Image(
                    label="False Color / NIR",
                    type="numpy",
                    height=320
                )

                swir_output = gr.Image(
                    label="SWIR Composite",
                    type="numpy",
                    height=320
                )


            # ------------------------------------------------
            # NDVI / NDWI / NDBI
            # ------------------------------------------------

            with gr.Row():

                ndvi_output = gr.Image(
                    label="NDVI",
                    type="numpy",
                    height=300
                )

                ndwi_output = gr.Image(
                    label="NDWI",
                    type="numpy",
                    height=300
                )

                ndbi_output = gr.Image(
                    label="NDBI",
                    type="numpy",
                    height=300
                )


            # ------------------------------------------------
            # NBR + UNCERTAINTY
            # ------------------------------------------------

            with gr.Row():

                nbr_output = gr.Image(
                    label="NBR",
                    type="numpy",
                    height=320
                )

                uncertainty_output = gr.Image(
                    label="LDSR-S2 Uncertainty",
                    type="numpy",
                    height=320
                )


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            zip_output = gr.File(
                label="Download All GeoTIFF Layers"
            )




    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    generate_button.click(

        fn=run_app,

        inputs=[
            latitude,
            longitude,
            start_date,
            end_date
        ],

        outputs=[
            comparison_output,
            false_color_output,
            swir_output,
            ndvi_output,
            ndwi_output,
            ndbi_output,
            nbr_output,
            uncertainty_output,
            zip_output
        ]
    )


print("✅ TerraVision UI + Interactive Map created successfully!")

demo.launch()
