import ollama
import tempfile
import os
from PIL import Image
from vlm_ifc import VLM


class OllamaVLM(VLM):
    def __init__(self, model_name="llama3.2-vision:11b"):
        self.model_name = model_name
    

    def generate_caption(self, image_source):
         
        print("Saving image...")
        tmp_file_path = "/tmp/image.jpg"  # Get the path to the temporary file
        image_source.save(tmp_file_path, format="JPEG")

        # Send the image to Ollama's model for analysis
        print("Starting Ollama chat...")
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': 'Caption this image in max 10 words.',
                'images': [tmp_file_path]
            }]
        )
        
        print(response)
            
        return response.get("message", {}).get("content", "")
        

