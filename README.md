# Stable Diffusion

Example taken from [Stable Diffision Keras Example](https://keras.io/examples/generative/random_walks_with_stable_diffusion/)

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
# Run with the default prompts
python main.py

# Run with another parameter set
#   - Extract the parameter file format
onecode-extract params.json

#   - Edit the prompts in the params.json file
#   - Run with it
python main.py params.json
```
