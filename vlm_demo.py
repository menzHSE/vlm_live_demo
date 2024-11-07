import cv2
import numpy as np
import subprocess
import argparse
import time  # Import time to add delay
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import threading
import queue  # Import queue for managing caption tasks
import llava_ifc  # Import llava_ifc only after threading has been initialized
import ollama_ifc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global queue to hold captions and a variable to control audio output
caption_queue = queue.Queue(maxsize=1)  # Limit the queue to 1 element to always keep the newest caption
audio_enabled = False  # Track whether audio output is enabled


# Function to display available key commands
def display_help():
    print("\nAvailable Key Commands:")
    print("'q' : Quit the application.")
    print("'c' : Toggle captioning on/off.")
    print("'a' : Toggle audio on/off.")
    print("'h' : Display this help message.\n")

# Function to run macOS 'say' command in a separate thread and speak captions sequentially
def speak_caption_worker():
    while True:
        # Get the next caption from the queue
        caption = caption_queue.get()
        if caption is None:
            break  # Exit the worker thread
        
        # Speak the caption using 'say' command only if audio is enabled
        if audio_enabled:
            subprocess.run(['say', caption])
        
        # Mark the task as done
        caption_queue.task_done()

# Function to add a new caption to the queue (replace the old one if the queue is full)
def add_caption_to_queue(caption):
    if not caption_queue.empty():
        try:
            caption_queue.get_nowait()  # Remove the oldest caption
        except queue.Empty:
            pass  # In case there is no caption, do nothing
    caption_queue.put(caption)

# Function to initialize TTS and start threading
def init_tts_engine():
    # Start the caption worker thread
    worker_thread = threading.Thread(target=speak_caption_worker, daemon=True)
    worker_thread.start()
    return None  # No need for further initialization

# Function to split the text into multiple lines that fit within the image width
def wrap_text(text, font, font_scale, thickness, img_width):
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        # Calculate the width of the current line if we add this word
        text_size, _ = cv2.getTextSize(current_line + word, font, font_scale, thickness)
        line_width = text_size[0]

        # If the current line width exceeds the image width, start a new line
        if line_width > img_width - 20:  # 20 is the padding
            lines.append(current_line)
            current_line = word + " "  # Start a new line with the current word
        else:
            current_line += word + " "
    
    # Append the last line
    if current_line:
        lines.append(current_line.strip())
    
    return lines

# Function to list available cameras using FFmpeg
def list_available_cameras():
    result = subprocess.run(
        ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    output = result.stderr
    video_devices = {}
    is_video_section = False
    for line in output.split('\n'):
        if 'AVFoundation video devices' in line:
            is_video_section = True
            continue
        if 'AVFoundation audio devices' in line:
            break
        if is_video_section and '[' in line and ']' in line:
            try:
                bracket_pairs = [i for i in range(len(line)) if line[i] == '[']
                if len(bracket_pairs) >= 2:
                    index = int(line[bracket_pairs[1] + 1:line.find(']', bracket_pairs[1])])
                    device_name = line[line.find(']') + 1:].strip()
                    video_devices[index] = device_name
            except ValueError:
                continue

    return video_devices

# Function to add transparent rectangle with multiline text at the bottom of the frame
def add_caption_to_frame(frame, text="Placeholder caption", font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=1):
    img_height, img_width, _ = frame.shape
    lines = wrap_text(text, font, font_scale, thickness, img_width)

    text_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1]
    line_spacing = 5
    total_text_height = len(lines) * (text_height + line_spacing)

    rect_x1, rect_y1 = 0, img_height - total_text_height - 50
    rect_x2, rect_y2 = img_width, img_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y_offset = img_height - total_text_height - 15
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_width = text_size[0]
        text_x = (img_width - text_width) // 2
        cv2.putText(frame, line, (text_x, y_offset), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y_offset += text_height + line_spacing
    
    return frame

# Function to generate a caption using LLava
def generate_caption(model, frame):
    
    # Convert OpenCV frame to PIL Image for the VLM
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Generate caption
    result = model.generate_caption(pil_image)
    return result

# Function to continuously capture and display images with optional LLava caption and TTS
def capture_and_display_continuous(camera_index, model, tts_engine, caption_interval):
    global audio_enabled
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open the camera at index {camera_index}.")
        return

    captioning_enabled = False  
    latest_caption = "No caption available"  
    latest_frame = None  
    executor = ThreadPoolExecutor(max_workers=1)
    caption_future = None  

    def update_caption_in_background(model, frame):
        nonlocal latest_caption
        caption = generate_caption(model, frame)
        latest_caption = caption
        add_caption_to_queue(caption)

    last_caption_time = time.time()  # Track the last time a caption was updated

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read the frame.")
                continue

            latest_frame = frame

            # Update caption based on specified interval           
            current_time = time.time()
            if captioning_enabled and (current_time - last_caption_time >= caption_interval):
                if caption_future is None or caption_future.done():
                    caption_future = executor.submit(update_caption_in_background, model, latest_frame)
                    print(f"Caption done in {(current_time - last_caption_time):4.2} seconds.")
                    last_caption_time = current_time
                    

            
            frame_with_caption = add_caption_to_frame(frame, text=latest_caption) if captioning_enabled else frame
            cv2.imshow('Webcam Feed with Caption', frame_with_caption)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting the loop.")
                break
            elif key == ord('c'):
                captioning_enabled = not captioning_enabled
                print(f"Captioning {'enabled (interval: ' + str(caption_interval) + ' sec)' if captioning_enabled else 'disabled'}.")
            elif key == ord('a'):
                audio_enabled = not audio_enabled
                print(f"Audio {'enabled' if audio_enabled else 'disabled'}.")
            elif key == ord('h'):
                display_help()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown()

# Command-line interface to select the camera by index and frame rate
def main():
    parser = argparse.ArgumentParser(description="Live Demo of Image Captioning with VLMs")
    parser.add_argument("-i", "--index", type=int, required=True, help="Index of the camera to use")
    parser.add_argument("-m", "--model", type=str, required=False, default="llava", help="Name of the model to use")
    parser.add_argument("-c", "--caption_interval", type=int, required=False, default=0, help="Interval in seconds between caption updates")
    parser.add_argument("--mlx", action="store_true", required=False, default=False, help="Use the mlx version of LLava")

    args = parser.parse_args()
    camera_index = args.index
    model_name = args.model
    caption_interval = args.caption_interval
   
    
    devices = list_available_cameras()
    print("Available cameras:")
    for index, name in devices.items():
        print(f"[{index}] {name}")

    if camera_index not in devices:
        print(f"Error: No camera found with index {camera_index}")
        return

    # Initialize TTS engine (macOS 'say' does not need initialization)
    tts_engine = init_tts_engine()

    if args.mlx:
        # Load LLava model and processor once, after threading is initialized
        sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../mlx-examples/llava')))
        model_name = "llava-hf/llava-1.5-7b-hf"
        model = llava_ifc.LLavaMLX(model_name, {})
    else:
        # Ollama interface
        model = ollama_ifc.OllamaVLM(model_name)
        
    print(f"Using model: {model_name}")

    # Start capturing from the selected camera at the specified frame rate
    capture_and_display_continuous(camera_index, model, tts_engine, caption_interval)

    # Clean up the model after use
    del model

    # Stop the worker thread
    add_caption_to_queue(None)  # Signal the worker to stop


if __name__ == "__main__":
    main()
