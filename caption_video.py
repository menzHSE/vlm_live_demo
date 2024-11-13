import cv2
import ollama
import tempfile
import os
from PIL import Image
from vlm_ifc import VLM
from ollama_ifc import OllamaVLM


def process_video(video_path, output_dir, model_name="llava", frame_interval=1, prompt="Describe this image in a short single sentence. Please do not exceed 15 words in total."):
    # Initialize Ollama VLM model
    vlm_model = OllamaVLM(model_name=model_name)
    
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the prompt to a text file in the output directory
    prompt_path = os.path.join(output_dir, "prompt.txt")
    with open(prompt_path, "w") as prompt_file:
        prompt_file.write(prompt)

    # Load video with OpenCV
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    success = True

    while success:
        # Read frame-by-frame
        success, frame = video_capture.read()
        
        if success:
            if frame_count % frame_interval == 0:
                # Convert frame to PIL image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Save the frame as an image
                image_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                pil_image.save(image_path)
                
                # Generate caption for the frame using the custom prompt
                caption = vlm_model.generate_caption(pil_image, prompt=prompt)
                
                # Save caption in a text file
                caption_path = os.path.join(output_dir, f"frame_{frame_count:04d}.txt")
                with open(caption_path, "w") as caption_file:
                    caption_file.write(caption)
                
                print(f"Processed frame {frame_count} with caption: {caption}")
            frame_count += 1

    # Release the video capture object
    video_capture.release()
    print("Processing complete.")


if __name__ == "__main__":
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process video to generate captions for each frame.")
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Path to the MP4 video file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory where images and captions will be saved.")
    parser.add_argument("-m", "--model", type=str, required=False, default="llava", help="Name of the model to use")
    parser.add_argument("-f", "--frame_interval", type=int, required=False, default=1, help="Process every n-th frame for caption generation")
    parser.add_argument("-p", "--prompt", type=str, required=False, default="Describe this image in a short single sentence. Please do not exceed 15 words in total.", help="Prompt to use for generating captions")

    args = parser.parse_args()
    
    # Process the video
    process_video(args.video_path, args.output_dir, model_name=args.model, frame_interval=args.frame_interval, prompt=args.prompt)
