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
    model = keras_cv.models.StableDiffusion(jit_compile=True)

    prompt_1 = onecode.text_input("Prompt 1", "A watercolor painting of a Golden Retriever at the beach")
    prompt_2 =  onecode.text_input("Prompt 2", "A still life DSLR photo of a bowl of fruit")
    interpolation_steps = onecode.slider("Interpolaion Steps", 5, min=3, max=500, step=1)

    encoding_1 = ops.squeeze(model.encode_text(prompt_1))
    encoding_2 = ops.squeeze(model.encode_text(prompt_2))

    interpolated_encodings = ops.linspace(encoding_1, encoding_2, interpolation_steps)

    # show the size of the latent manifold
    onecode.Logger.debug(f"Encoding shape: {encoding_1.shape}")

    # keep the diffusion noise constant between images.
    noise = keras.random.normal((512 // 8, 512 // 8, 4), seed=404)

    onecode.Logger.info("Generate images...")
    images = model.generate_image(
        interpolated_encodings,
        batch_size=interpolation_steps,
        diffusion_noise=noise,
    )

    output_filename = onecode.text_input("GIF Name", "Watercolor_fruits.gif")

    export_as_gif(
        onecode.file_output("gif", output_filename),
        [Image.fromarray(img) for img in images],
        frames_per_second=2,
        rubber_band=True,
    )
