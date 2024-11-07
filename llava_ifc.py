import codecs
from pathlib import Path
import sys
import os
import mlx
import mlx.core as mx
import requests
from PIL import Image
from transformers import AutoProcessor
from vlm_ifc import VLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'mlx/mlx-examples/llava')))
from llava import LlavaModel  # type: ignore

class LLavaMLX(VLM):
    def __init__(self, model_path, tokenizer_config={}):
        """
        Initialize the LLavaMLX with a model and processor.
        """
        self.processor = AutoProcessor.from_pretrained(model_path, **tokenizer_config)
        self.model = LlavaModel.from_pretrained(model_path)

    @staticmethod
    def load_image(image_source):
        """
        Load an image from a URL or file.
        """
        if image_source.startswith(("http://", "https://")):
            try:
                response = requests.get(image_source, stream=True)
                response.raise_for_status()
                return Image.open(response.raw)
            except Exception as e:
                raise ValueError(f"Failed to load image from URL: {image_source} with error {e}")
        elif Path(image_source).is_file():
            try:
                return Image.open(image_source)
            except IOError as e:
                raise ValueError(f"Failed to load image {image_source} with error: {e}")
        else:
            raise ValueError(f"The image {image_source} must be a valid URL or existing file.")

    def prepare_inputs(self, image, prompt):
        """
        Prepare inputs for the model using the processor.
        """
        if isinstance(image, str):
            image = self.load_image(image)
        inputs = self.processor(prompt, image, return_tensors="np")
        pixel_values = mx.array(inputs["pixel_values"])
        input_ids = mx.array(inputs["input_ids"])
        return input_ids, pixel_values

    @staticmethod
    def sample(logits, temperature=0.0):
        """
        Sample from the logits with optional temperature scaling.
        """
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temperature))

    def generate_text(self, input_ids, pixel_values, max_tokens, temperature):
        """
        Generate text based on image and prompt.
        """
        logits, cache = self.model(input_ids, pixel_values)
        logits = logits[:, -1, :]
        y = self.sample(logits, temperature=temperature)
        tokens = [y.item()]

        for _ in range(max_tokens - 1):
            logits, cache = self.model.language_model(y[None], cache=cache)
            logits = logits[:, -1, :]
            y = self.sample(logits, temperature)
            token = y.item()
            if token == self.processor.tokenizer.eos_token_id:
                break
            tokens.append(token)

        return self.processor.tokenizer.decode(tokens)
    
    def generate_caption(self, image_source):
        
        # Parameters for caption generation
        prompt_template = "USER: <image>\nDescribe this image in a single sentence.\nASSISTANT:"
        max_tokens = 32
        temp = 0.5
        
        return self.generate_caption_internal(image_source, prompt_template, max_tokens, temp)


    def generate_caption_internal(self, image_source, prompt, max_tokens=100, temperature=0.3):
        """
        Generate a caption for an image with a given prompt.
        """
     
        prompt = codecs.decode(prompt, "unicode_escape")
        input_ids, pixel_values = self.prepare_inputs(image_source, prompt)
        return self.generate_text(input_ids, pixel_values, max_tokens, temperature)


# Example usage
if __name__ == "__main__":
    model_path = "llava-hf/llava-1.5-13b-hf"
    image_source = "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
    max_tokens = 128
    temperature = 0.3

    # Initialize the generator and generate a caption
    generator = LLavaMLX(model_path)
    result = generator.generate_caption(image_source, prompt, max_tokens, temperature)
    print(result)
