import codecs
from pathlib import Path

import mlx
import mlx.core as mx
import requests
from PIL import Image
from transformers import AutoProcessor

from llava import LlavaModel # type: ignore


def load_image(image_source):
    """
    Helper function to load an image from either a URL or file.
    """
    if image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            )
    elif Path(image_source).is_file():
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image {image_source} with error: {e}")
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )


def prepare_inputs(processor, image, prompt):
    if isinstance(image, str):
        image = load_image(image)
    inputs = processor(prompt, image, return_tensors="np")
    pixel_values = mx.array(inputs["pixel_values"])
    input_ids = mx.array(inputs["input_ids"])
    return input_ids, pixel_values


def load_model(model_path, tokenizer_config={}):
    processor = AutoProcessor.from_pretrained(model_path, **tokenizer_config)
    model = LlavaModel.from_pretrained(model_path)
    return processor, model


def sample(logits, temperature=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


def generate_text(input_ids, pixel_values, model, processor, max_tokens, temperature):
    logits, cache = model(input_ids, pixel_values)
    logits = logits[:, -1, :]
    y = sample(logits, temperature=temperature)
    tokens = [y.item()]

    for n in range(max_tokens - 1):
        logits, cache = model.language_model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits, temperature)
        token = y.item()
        if token == processor.tokenizer.eos_token_id:
            break
        tokens.append(token)

    return processor.tokenizer.decode(tokens)


def generate_caption(processor, model, image_source, prompt, max_tokens=100, temp=0.3, eos_token=None):
    """
    Function to generate caption or response for an image using a model.

    Parameters:
        model_name (str): The path to the model or Hugging Face repository.
        image_source (str): URL or path to the image.
        prompt (str): The prompt text to generate responses for.
        max_tokens (int): Maximum number of tokens to generate.
        temp (float): Temperature for sampling (controls randomness).
        eos_token (str, optional): End of sequence token for the tokenizer.

    Returns:
        str: Generated text based on the image and prompt.
    """


    # Decode prompt if necessary
    prompt = codecs.decode(prompt, "unicode_escape")

    # Prepare inputs for the model
    input_ids, pixel_values = prepare_inputs(processor, image_source, prompt)

    # Generate text using the model
    generated_text = generate_text(
        input_ids, pixel_values, model, processor, max_tokens, temp
    )
    
    return generated_text

# Example usage:
if __name__ == "__main__":
    # Example parameters
    model_name = "llava-hf/llava-1.5-13b-hf"
    image_source = "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
    max_tokens = 128
    temp = 0.3

    # Generate caption using the callable function
    result = generate_caption(model_name, image_source, prompt, max_tokens, temp)
    print(result)
