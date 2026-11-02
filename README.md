# ROS Package [bob_flux3k](https://github.com/bob-ros2/bob_flux3k)

ROS 2 node for the **FLUX.3 Image Klein** text-to-image and multi-reference image-to-image models from Black Forest Labs.

## Supported Models

This node supports both open FLUX.3 Image Klein checkpoints:
- **[flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)** (default): Step-distilled, 16 denoising steps, no classifier-free guidance (Apache-style fast inference, FLUX Kommunity License).
- **[flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)**: Base model, 50 steps with CFG (recommended `guidance_scale` 3.5). Non-commercial FLUX Kommunity License.

> [!IMPORTANT]
> Both checkpoints are **gated**: visit the model page, log in to Hugging Face and accept the FLUX Kommunity License. The node needs a logged-in token (`huggingface-cli login` or the `HF_TOKEN` environment variable) to download the weights.

You can switch models via the `repo_id` ROS parameter or the `FLUX3K_REPO_ID` environment variable.

## Installation & Build

### 1. Clone the repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/bob-ros2/bob_flux3k.git
```

### 2. Install dependencies
It is recommended to use a virtual environment.
```bash
pip install -r bob_flux3k/requirements.txt
```
> FLUX.3 ships as a *modular* diffusers pipeline. Until a diffusers release contains Flux3 support, `requirements.txt` tracks `diffusers` from git main.

### 3. Build the workspace
```bash
cd ~/ros2_ws
colcon build --packages-select bob_flux3k
source install/setup.bash
```

## Usage

### Run the node
You can run the node directly or provide a configuration file.

**Basic run:**
```bash
ros2 run bob_flux3k tti --ros-args -p repo_id:=black-forest-labs/flux-3-image-klein
```

**Using a configuration file:**
```bash
ros2 run bob_flux3k tti --ros-args --params-file src/bob_flux3k/config/tti.yaml
```

### Send a prompt
```bash
ros2 topic pub /flux3k/prompt std_msgs/msg/String "{data: 'A futuristic city in the style of cyberpunk'}" --once
```

> [!NOTE]
> FLUX.3 Image Klein takes **full scene captions** (subject, composition, lighting, style). Short prompts produce generic output. Black Forest Labs ships a separate 9B prompt upsampler (`flux-3-image-klein-pup`); set `use_prompt_upsampler: true` to run it inside the pipeline (~19 GB extra download and VRAM).

### JSON Prompt Support (Dynamic ITI/TTI)
The node automatically detects if a prompt is plain text or a JSON string. Using JSON allows you to specify input image(s) dynamically for a single request. **Unknown JSON keys are logged as a warning and ignored**; processing continues.

**Format:**
```json
{
  "content": "A high-quality photo of a cat",
  "image_url": "file:///path/to/image.jpg"
}
```

**Multi-reference image-to-image (FLUX.3 feature)** — pass a list; refer to the images in the caption as `<ref_image_0>`, `<ref_image_1>`, ...:
```json
{
  "content": "The cat from <ref_image_0> sitting on the bench from <ref_image_1>.",
  "image_url": ["file:///tmp/cat.jpg", "file:///tmp/bench.jpg"]
}
```

**Features:**
- **Text-to-Image**: Just send plain text or JSON without an `image_url`.
- **Image-to-Image**: Provide an `image_url` (string or list of strings). Supports:
    - Local files: `file:///home/user/image.png`
    - Remote URLs: `https://example.com/image.jpg`
    - **Base64 encoded**: `data:image/png;base64,...` (Ideal for web-app integrations).
- **Layout control**: append a JSON list of regions to `content` (`id`, `bbox` as `[y_min, x_min, y_max, x_max]` on a 0–1000 grid, `desc`), see the [FLUX.3 model card](https://huggingface.co/black-forest-labs/flux-3-image-klein).
- **Per-request parameter overrides**: the dynamic parameters can be overridden for a single request directly in the JSON body; wrong types are ignored with a warning and the ROS parameter values apply:

| JSON key | Type | Replaces ROS parameter |
| :--- | :--- | :--- |
| `seed` | int | `seed` (`-1` = random) |
| `num_inference_steps` | int | `num_inference_steps` |
| `guidance_scale` | number | `guidance_scale` |
| `num_images_per_prompt` | int | `num_images_per_prompt` |
| `height` / `width` | int | `height` / `width` |

**Example:**
```json
{
  "content": "A high-quality photo of a cat",
  "num_inference_steps": 8,
  "guidance_scale": 2.5,
  "seed": 42,
  "height": 768,
  "width": 1024
}
```

**Example (CLI):**
```bash
ros2 topic pub /flux3k/prompt std_msgs/msg/String "{data: '{\"content\": \"a robot dog\", \"image_url\": \"file:///tmp/dog.jpg\"}'}" --once
```

## ROS API

### Topics

The node interacts with the following ROS 2 topics:

| Topic | Message Type | Direction | Description |
| :--- | :--- | :--- | :--- |
| `/flux3k/prompt` | `std_msgs/msg/String` | **Sub** | The input prompt for text-to-image or image-conditioned generation. |
| `/flux3k/generated_image` | `sensor_msgs/msg/Image` | **Pub** | The resulting image, published in `bgr8` encoding. |

### Parameters

#### Dynamic Parameters
These can be adjusted while the node is running using the **`rqt_reconfigure`** GUI.

| Parameter | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `keep_loaded` | `FLUX3K_KEEP_LOADED` | `false` | If true, keeps the model in memory. |
| `cpu_offload` | `FLUX3K_CPU_OFFLOAD` | `true` | If true, uses automatic component CPU offloading (`ComponentsManager`). |
| `num_inference_steps` | `FLUX3K_NUM_STEPS` | `16` | Denoising steps (distilled: 4–16; base: 50). |
| `guidance_scale` | `FLUX3K_GUIDANCE` | `3.5` | CFG strength. Only checkpoints with a guider component (base model) use it; ignored by the distilled model. |
| `num_images_per_prompt`| `FLUX3K_NUM_IMAGES` | `1` | Batch generation count. |
| `frame_id` | `FLUX3K_FRAME_ID` | `flux3k` | The frame_id for the output. |

#### Static Parameters
These are set at **startup** and cannot be changed while the node is running.

| Parameter | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `repo_id` | `FLUX3K_REPO_ID` | `black-forest-labs/flux-3-image-klein` | Model repository ID. |
| `model_dir` | `FLUX3K_MODEL_DIR` | `./models` | Cache directory. |
| `device` | `FLUX3K_DEVICE` | `cuda:0` | Computing device. |
| `seed` | `FLUX3K_SEED` | `-1` | Random seed (-1 for random). |
| `image_path` | `FLUX3K_IMAGE_PATH` | `''` | Path for saving the output image. If empty, local saving is disabled. |
| `once` | `FLUX3K_ONCE` | `false` | Exit after first generation if true. |
| `height` | `FLUX3K_HEIGHT` | `1024` | Image height. |
| `width` | `FLUX3K_WIDTH` | `1024` | Image width. |
| `use_prompt_upsampler` | `FLUX3K_USE_UPSAMPLER` | `false` | Run the bundled 9B prompt upsampler inside the pipeline. |

### Dynamic Reconfiguration GUI

To start the configuration GUI and adjust the dynamic parameters, run:
```bash
ros2 run rqt_reconfigure rqt_reconfigure
```

## Docker

The package contains a `Dockerfile`, `entrypoint.sh` and `docker-compose.yaml`. Models are cached **inside the repository** under `models/` (gitignored) and mounted into the container; generated images land in `generated_images/`.

```bash
cd ros2_ws/src/bob_flux3k
cp .env.example .env         # put your HF_TOKEN here (or export HF_TOKEN)
docker compose up -d --build
docker logs -f bob_flux3k    # first start downloads the model into ./models
```

> The container runs as `user: 1000:1000` (the host user `ros`) so the FastDDS shared-memory transport works with host processes (`ipc: host`). For a different host user, set `FLUX3K_UID`/`FLUX3K_GID` in `.env`.

Send a prompt from the host (shares the host network and `ROS_DOMAIN_ID`):
```bash
ros2 topic pub /flux3k/prompt std_msgs/msg/String "{data: 'A futuristic city in the style of cyberpunk'}" --once
```

## Configuration File

An example configuration file is provided at `config/tti.yaml`. This file contains all parameters with descriptions and can be used to set the initial behavior of the node without providing long command-line arguments.

## Hardware Optimization

The node is optimized for NVIDIA consumer GPUs (like the RTX 4090) using `torch.bfloat16`. The distilled checkpoint at 1024×1024 peaks around 21 GB VRAM at the last decoding steps; with `cpu_offload: true` (default) components are moved to the GPU only while they run, allowing it to coexist with other GPU-intensive nodes.

> **Multiple image nodes on one GPU:** the default is `keep_loaded: false` with `cpu_offload: true` — the weights are paged onto the GPU per request on demand and released completely afterwards (without offloading, a 1024² generation with ~19.3 GB of weights does not reliably fit in 24 GB). Two generations running at the same time still do not fit in 24 GB — drive requests sequentially. Topics live under `/flux3k/…` so the nodes never trigger each other.
