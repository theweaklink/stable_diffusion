import onecode
import keras_cv
import keras
import matplotlib.pyplot as plt
from keras import ops
import numpy as np
import math
from PIL import Image

from .gif_export import export_as_gif


def run():
    onecode.Logger.info("Instantiating the Stable Diffusion model...")
    size = onecode.slider('Image Size', 256, min=128, max=512, step=128)
    model = keras_cv.models.StableDiffusion(img_height=size, img_width=size, jit_compile=False)

    prompt_1 = onecode.text_input("Prompt 1", "A whale in outer space")
    prompt_2 =  onecode.text_input("Prompt 2", "An astronaut riding a horse")
    interpolation_steps = onecode.slider("Interpolation Steps", 8, min=3, max=100, step=1)

    encoding_1 = ops.squeeze(model.encode_text(prompt_1))
    encoding_2 = ops.squeeze(model.encode_text(prompt_2))

    interpolated_encodings = ops.linspace(encoding_1, encoding_2, interpolation_steps)

    # show the size of the latent manifold
    onecode.Logger.debug(f"Encoding shape: {encoding_1.shape}")

    # keep the diffusion noise constant between images.
    noise = keras.random.normal((size // 8, size // 8, 4), seed=404)

    onecode.Logger.info("Generate images...")
    images = model.generate_image(
        interpolated_encodings,
        batch_size=interpolation_steps,
        diffusion_noise=noise if onecode.checkbox('Use noise?', False) else None,
        num_steps=onecode.slider("ML iterations", 50, min=10, max=100, step=1),
        unconditional_guidance_scale=onecode.number_input("Guidance scale", 7, min=1, step=1)
    )

    onecode.Logger.info("Exporting to GIF...")
    output_filename = onecode.text_input("GIF Name", "Space_Whale.gif")
    export_as_gif(
        onecode.file_output("gif", output_filename),
        [Image.fromarray(img) for img in images],
        frames_per_second=onecode.slider('GIF FPS', 2, min=1, max=30, step=1),
        rubber_band=onecode.checkbox("GIF Rubberband?", False),
    )
