import ollama
import tempfile
import os
from PIL import Image
from vlm_ifc import VLM


class OllamaVLM(VLM):
    def __init__(self, model_name="llava"):
        self.model_name = model_name
    

    def generate_caption(self, image_source, prompt="Describe this image in a short single sentence. Please do not exceed 15 words in total."):
        # Create a temporary file in a context manager
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp_file:
            # Save the image to the temporary file
            image_source.save(tmp_file.name, format="JPEG")
            
            # Ensure the file is flushed so it can be read
            tmp_file.flush()
            
            # Send the image to Ollama's model for analysis
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [tmp_file.name]
                }],
                options={'temperature': 0.3}
            )
        
        # Return the caption from the response    
        return response.get("message", {}).get("content", "")
        

