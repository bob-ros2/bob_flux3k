import os
import sys
import time
import random
import string
import gc
import json
import base64
import io

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import ParameterType

import torch
# Set allocator configuration before any CUDA work to prevent fragmentation
if 'PYTORCH_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import load_image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PILImage

"""
ROS node for the **FLUX.3 Image Klein** text-to-image and multi-reference
image-to-image models from Black Forest Labs.
"""

class Flux3Knode(Node):
    """
    A ROS node that interfaces with the FLUX.3 Image Klein model for text-to-image,
    multi-reference image-to-image generation and bounding-box layout control.
    """

    def __init__(self):
        """
        Initializes the Flux3Knode, declaring parameters, creating subscribers/publishers,
        and setting up the image processing bridge.
        """
        super().__init__('tti')
        self.get_logger().info("Initializing Flux3-klein ROS node...")

        self.bridge = CvBridge()

        # --- ROS Parameter Declaration ---
        self.declare_parameter(
            'repo_id',
            os.environ.get('FLUX3K_REPO_ID', 'black-forest-labs/flux-3-image-klein'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The Huggingface repo id to use.'
            )
        )
        self.declare_parameter(
            'model_dir',
            os.environ.get('FLUX3K_MODEL_DIR', os.path.join(os.getcwd(), 'models')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Directory to cache models.'
            )
        )
        self.declare_parameter(
            'device',
            os.environ.get('FLUX3K_DEVICE', 'cuda:0'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The torch device to use.'
            )
        )
        self.declare_parameter(
            'prompt',
            os.environ.get('FLUX3K_PROMPT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='An initial prompt to generate an image at startup.'
            )
        )
        self.declare_parameter(
            'input_image',
            os.environ.get('FLUX3K_INPUT_IMAGE', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path to an input image to condition the generation.'
            )
        )
        self.declare_parameter(
            'once',
            os.environ.get('FLUX3K_ONCE', 'false').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, the node will exit after generating the first image.'
            )
        )
        self.declare_parameter(
            'image_path',
            os.environ.get('FLUX3K_IMAGE_PATH', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path for saving the output image. If empty, local saving is disabled.'
            )
        )
        self.declare_parameter(
            'seed',
            int(os.environ.get('FLUX3K_SEED', -1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Seed for the random number generator.'
            )
        )
        self.declare_parameter(
            'image_counter_start',
            int(os.environ.get('FLUX3K_IMAGE_COUNTER_START', 1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='The starting value for the image filename counter.'
            )
        )
        self.declare_parameter(
            'height',
            int(os.environ.get('FLUX3K_HEIGHT', 1024)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Height of the generated image.'
            )
        )
        self.declare_parameter(
            'width',
            int(os.environ.get('FLUX3K_WIDTH', 1024)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Width of the generated image.'
            )
        )
        self.declare_parameter(
            'num_inference_steps',
            int(os.environ.get('FLUX3K_NUM_STEPS', 16)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Number of denoising steps (distilled: 4-16, default 16; base: 50).'
            )
        )
        self.declare_parameter(
            'guidance_scale',
            float(os.environ.get('FLUX3K_GUIDANCE', 3.5)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Classifier-free guidance strength. Only used by checkpoints with a guider component (base model); ignored by the step-distilled model.'
            )
        )
        self.declare_parameter(
            'num_images_per_prompt',
            int(os.environ.get('FLUX3K_NUM_IMAGES', 1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Number of images to generate per prompt.'
            )
        )
        self.declare_parameter(
            'frame_id',
            os.environ.get('FLUX3K_FRAME_ID', 'flux3k'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The frame_id for the generated image messages.'
            )
        )
        self.declare_parameter(
            'keep_loaded',
            os.environ.get('FLUX3K_KEEP_LOADED', 'true').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, keeps the model in memory between prompts.'
            )
        )
        self.declare_parameter(
            'cpu_offload',
            os.environ.get('FLUX3K_CPU_OFFLOAD', 'true').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, uses automatic component CPU offloading (ComponentsManager) to save VRAM.'
            )
        )
        self.declare_parameter(
            'use_prompt_upsampler',
            os.environ.get('FLUX3K_USE_UPSAMPLER', 'false').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, runs the bundled prompt upsampler (9B "pup" model, ~19GB extra download and VRAM) to expand short requests into full scene captions.'
            )
        )

        # --- Get Parameters ---
        self.repo_id = self.get_parameter('repo_id').value
        self.model_dir = self.get_parameter('model_dir').value
        self.device = self.get_parameter('device').value
        self.image_path = self.get_parameter('image_path').value
        self.prompt = self.get_parameter('prompt').value
        self.seed = self.get_parameter('seed').value
        self.input_image = self.get_parameter('input_image').value
        self.once = self.get_parameter('once').value
        self.image_counter = self.get_parameter('image_counter_start').value
        self.height = self.get_parameter('height').value
        self.width = self.get_parameter('width').value
        self.torch_dtype = torch.bfloat16
        self.pipe = None
        self.current_repo_id = None
        self.current_cpu_offload = None
        self.current_use_upsampler = None
        self.current_guidance = None

        # Ensure model_dir exists
        if self.model_dir and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            self.get_logger().info(f"Created model directory: {self.model_dir}")

        # Ensure image_path directory exists if using auto
        if 'auto' in self.image_path:
            out_dir = os.path.dirname(self.image_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        self.get_logger().info(f"Using repo_id: '{self.repo_id}'")
        self.get_logger().info(f"Using device: '{self.device}'")

        # --- Subscriber ---
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            1000)

        # --- Publisher ---
        self.image_pub = self.create_publisher(Image, 'generated_image', 10)

        # --- Handle initial prompt ---
        if self.prompt:
            self.get_logger().info("Processing initial prompt...")
            # Use Timer to avoid blocking constructor/initialization
            timer = self.create_timer(0.1, lambda: self._initial_prompt_timer_callback(timer))

    def _initial_prompt_timer_callback(self, timer):
        """
        One-shot timer callback to process the initial prompt after the node has initialized.
        """
        timer.cancel()
        self.prompt_callback(String(data=self.prompt))

    # Per-request parameter overrides accepted inside the JSON prompt body.
    # All of them can be changed while the model is loaded; wrong types are
    # ignored with a warning and fall back to the ROS parameter values.
    REQUEST_PARAM_TYPES = {
        'seed': int,
        'num_inference_steps': int,
        'guidance_scale': float,
        'num_images_per_prompt': int,
        'height': int,
        'width': int,
    }

    def _parse_prompt(self, msg_data):
        """
        Parses the incoming prompt message.
        Supports plain text or a JSON string like:
        {"content": "...", "image_url": "..."} or
        {"content": "...", "image_url": ["...", "..."]} for multiple reference images.
        Unknown JSON keys are logged as a warning and ignored.
        Recognized request parameters (see REQUEST_PARAM_TYPES) override the
        ROS parameter values for this single request.
        Returns (content, image_url, overrides).
        """
        overrides = {}
        try:
            data = json.loads(msg_data)
            if isinstance(data, dict):
                if 'content' not in data:
                    self.get_logger().error("JSON prompt missing 'content' key.")
                    return None, None, overrides
                supported_keys = {'content', 'image_url'} | set(self.REQUEST_PARAM_TYPES)
                unsupported_keys = [key for key in data.keys() if key not in supported_keys]
                if unsupported_keys:
                    self.get_logger().warning(
                        f"Unsupported JSON key(s) ignored: {', '.join(sorted(unsupported_keys))}")
                for key, value in data.items():
                    expected = self.REQUEST_PARAM_TYPES.get(key)
                    if expected is None:
                        continue
                    if expected is int:
                        if isinstance(value, bool) or not isinstance(value, int):
                            self.get_logger().warning(
                                f"JSON key '{key}' ignored: expected integer, got {type(value).__name__}.")
                            continue
                        if value < (0 if key != 'seed' else -1) or (key != 'seed' and value == 0):
                            self.get_logger().warning(f"JSON key '{key}' ignored: value out of range ({value}).")
                            continue
                    elif expected is float:
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                            self.get_logger().warning(
                                f"JSON key '{key}' ignored: expected non-negative number, got {value!r}.")
                            continue
                        value = float(value)
                    overrides[key] = value
                return data.get('content'), data.get('image_url'), overrides
        except json.JSONDecodeError:
            pass # Not a JSON string, treat as plain text

        return msg_data, None, overrides

    def _load_input_image(self, image_input):
        """
        Loads an image from a URL, local file path, or base64 data string.
        """
        try:
            if image_input.startswith('data:image'):
                # Handle base64
                self.get_logger().info("Loading image from base64 data...")
                header, encoded = image_input.split(",", 1)
                image_data = base64.b64decode(encoded)
                return PILImage.open(io.BytesIO(image_data)).convert("RGB")

            # Strip file:// prefix if present
            clean_path = image_input
            if image_input.startswith('file://'):
                clean_path = image_input[7:]

            if not clean_path.startswith(('http://', 'https://')):
                # Local file handling
                if not os.path.exists(clean_path):
                    self.get_logger().error(f"Local image file does not exist: {clean_path}")
                    return None
                self.get_logger().info(f"Loading local image from: {clean_path}")
                return PILImage.open(clean_path).convert("RGB")
            else:
                # Handle URL via diffusers utility
                self.get_logger().info(f"Loading image from URL: {clean_path}")
                return load_image(clean_path).convert("RGB")
        except Exception as e:
            self.get_logger().error(f"Failed to load input image: {e}")
            return None

    def _load_input_images(self, image_input):
        """
        Loads one or more reference images. FLUX.3 accepts a single image or a
        list of reference images for multi-reference image-to-image.
        Returns a single PIL image, a list of PIL images, or None.
        """
        image_urls = image_input if isinstance(image_input, list) else [image_input]
        images = []
        for image_url in image_urls:
            image = self._load_input_image(image_url)
            if image is not None:
                images.append(image)
            else:
                self.get_logger().warning(f"Skipping unloadable image input: {image_url}")
        if not images:
            return None
        return images[0] if len(images) == 1 else images

    def _get_image_output_path(self):
        """
        Generates the output path for the generated image.
        If image_path ends with 'auto', generates a unique filename with a counter and random suffix.
        """
        if self.image_path.endswith('auto'):
            directory = os.path.dirname(self.image_path)
            random_chars = string.ascii_letters + string.digits
            random_string = ''.join(random.choice(random_chars) for _ in range(10))
            filename = f"flux3k_{self.image_counter:04d}_{random_string}.png"
            return os.path.join(directory, filename)
        else:
            return self.image_path

    def _build_pipeline(self, repo_id, cpu_offload, use_prompt_upsampler):
        """
        Builds the FLUX.3 modular pipeline. Loads from cache_dir, optionally drops
        the prompt upsampler block, and either enables automatic component CPU
        offloading (ComponentsManager) or moves the whole pipeline to the device.
        """
        blocks = ModularPipeline.from_pretrained(
            repo_id,
            cache_dir=self.model_dir
        ).blocks

        if not use_prompt_upsampler and 'prompt_upsampler' in blocks.sub_blocks:
            self.get_logger().info("Dropping prompt upsampler block (use_prompt_upsampler=false)...")
            blocks.sub_blocks.pop('prompt_upsampler')

        manager = ComponentsManager() if cpu_offload else None
        pipe = blocks.init_pipeline(repo_id, components_manager=manager)
        # cache_dir must be passed to load_components as well, otherwise the
        # component weights bypass model_dir and land in the default HF cache.
        pipe.load_components(dtype=self.torch_dtype, cache_dir=self.model_dir)

        if manager is not None:
            self.get_logger().info("Optimization: Enabling automatic component CPU offload...")
            manager.enable_auto_cpu_offload(device='cuda')
        else:
            self.get_logger().info(f"Optimization: Moving whole model to {self.device}...")
            pipe.to(self.device)

        # Guidance is a component property, not a call argument. Only checkpoints
        # with classifier-free guidance (base model) carry a guider component.
        guider = getattr(pipe, 'guider', None)
        guidance = self.get_parameter('guidance_scale').value
        if guider is not None:
            pipe.guider = guider.new(guidance_scale=guidance)
            self.get_logger().info(f"Set guidance_scale: {guidance}")
        else:
            self.get_logger().info(
                "Checkpoint has no guider component (step-distilled); guidance_scale is ignored.")
        self.current_guidance = guidance

        return pipe

    def prompt_callback(self, msg):
        """
        Callback for the 'prompt' subscription.
        Parses the prompt (supports JSON for dynamic ITI), loads or clears the model as needed,
        runs the inference, saves the result, and publishes the image.
        """
        prompt_data = msg.data
        prompt, json_image_url, overrides = self._parse_prompt(prompt_data)

        if prompt is None:
            return # Error already logged in _parse_prompt

        self.get_logger().info(f"Received prompt")
        self.get_logger().debug(f"Prompt: {prompt}")
        if overrides:
            self.get_logger().info(f"Request parameter overrides: {overrides}")

        callback_start = time.time()

        try:
            # --- Model Loading / Caching Logic ---
            cpu_offload = self.get_parameter('cpu_offload').value
            use_prompt_upsampler = self.get_parameter('use_prompt_upsampler').value
            repo_id = self.repo_id # Static for now, but we check anyway

            # Reload if not loaded or if configuration changed
            if (self.pipe is None or
                self.current_repo_id != repo_id or
                self.current_cpu_offload != cpu_offload or
                self.current_use_upsampler != use_prompt_upsampler):

                if self.pipe is not None:
                    self.get_logger().info("Configuration changed or pipe exists, clearing old pipeline...")
                    del self.pipe
                    self.pipe = None
                    gc.collect()
                    torch.cuda.empty_cache()

                start_time = time.time()
                self.get_logger().info(f"Loading Flux3-klein pipeline ({repo_id})...")

                self.pipe = self._build_pipeline(repo_id, cpu_offload, use_prompt_upsampler)

                self.get_logger().info(f"Pipeline loaded in {time.time() - start_time:.2f}s")
                self.current_repo_id = repo_id
                self.current_cpu_offload = cpu_offload
                self.current_use_upsampler = use_prompt_upsampler
            else:
                self.get_logger().debug("Using cached pipeline.")

            # Seed handling
            current_seed = overrides.get('seed', self.seed)
            if current_seed == -1:
                current_seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator().manual_seed(current_seed)

            # Prepare call arguments
            # We fetch parameter values dynamically to support RQT reconfiguration,
            # with per-request JSON overrides taking precedence
            num_steps = overrides.get('num_inference_steps', self.get_parameter('num_inference_steps').value)
            guidance = overrides.get('guidance_scale', self.get_parameter('guidance_scale').value)
            num_images = overrides.get('num_images_per_prompt', self.get_parameter('num_images_per_prompt').value)
            height = overrides.get('height', self.height)
            width = overrides.get('width', self.width)

            # Guidance is only re-appliable while running for checkpoints with a guider
            guider = getattr(self.pipe, 'guider', None)
            if guider is not None and guidance != self.current_guidance:
                self.pipe.guider = guider.new(guidance_scale=guidance)
                self.current_guidance = guidance

            self.get_logger().info(
                f"Inference Params: steps={num_steps}, guidance={guidance:.2f}, num_images={num_images}, height={height}, width={width}, current_seed={current_seed}")

            call_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_steps,
                "num_images_per_prompt": num_images,
                "generator": generator
            }

            # Determine input image(s)
            # Priority: 1. JSON-provided image_url, 2. ROS-parameter input_image
            target_image_input = json_image_url if json_image_url else self.input_image

            if target_image_input:
                input_imgs = self._load_input_images(target_image_input)
                if input_imgs:
                    call_kwargs["image"] = input_imgs
                else:
                    self.get_logger().warning("Image input provided but failed to load. Falling back to text-to-image.")

            self.get_logger().info(f"Generating image...")
            gen_start = time.time()
            result = self.pipe(**call_kwargs, output="images")
            try:
                images = result["images"]
            except (TypeError, KeyError):
                images = result

            for i, image in enumerate(images):
                self.get_logger().info(f"Generation finished (Image {i+1}/{len(images)}) in {time.time() - gen_start:.2f}s")

                # Save image
                if self.image_path:
                    output_path = self._get_image_output_path()
                    image.save(output_path)
                    self.get_logger().info(f"Image saved to: {output_path}")

                    if self.image_path.endswith('auto'):
                        self.image_counter += 1

                # Publish image as BGR8 (Standard for ROS 2 vision nodes)
                # We convert PIL RGB -> Numpy -> BGR
                image_rgb = np.array(image.convert("RGB"))
                image_bgr = image_rgb[:, :, ::-1].copy() # Swaps R and B channels and ensures memory is contiguous

                image_msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
                image_msg.header.stamp = self.get_clock().now().to_msg()
                image_msg.header.frame_id = self.get_parameter('frame_id').value
                self.image_pub.publish(image_msg)

        except torch.cuda.OutOfMemoryError:
            self.get_logger().error("Error in prompt_callback: CUDA out of memory.")
            self.get_logger().error("TIP: Enable 'cpu_offload' or close other GPU apps (browsers, editors).")
            # Clear cache for next attempt
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            self.get_logger().error(f"Error in prompt_callback: {e}")

        finally:
            keep_loaded = self.get_parameter('keep_loaded').value

            if not keep_loaded:
                self.get_logger().info("Cleaning up (keep_loaded=false)...")
                if self.pipe is not None:
                    del self.pipe
                    self.pipe = None
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # Release VRAM so sibling GPU nodes can coexist on this card:
                # weights are moved back on demand by the offload manager on
                # the next request. This also restores a consistent placement
                # after a failed call.
                if self.pipe is not None:
                    for component_name in ('transformer', 'text_encoder', 'vae'):
                        try:
                            getattr(self.pipe, component_name).to('cpu')
                        except (AttributeError, RuntimeError):
                            pass
                    gc.collect()
                    torch.cuda.empty_cache()
                self.get_logger().debug("Keeping model loaded for next prompt (weights offloaded to CPU).")

            if self.once:
                self.get_logger().info("'once' is True, exiting node.")
                sys.exit(0)

            total_duration = time.time() - callback_start
            self.get_logger().info(f"DONE: Total callback duration: {total_duration:.2f}s")

def main(args=None):
    """
    Main entry point for the Flux3K ROS 2 node.
    """
    rclpy.init(args=args)
    node = Flux3Knode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
