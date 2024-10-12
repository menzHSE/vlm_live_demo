import cv2
import numpy as np
import subprocess
import argparse
import time  # Import time to add delay

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
    # Run the ffmpeg command to list avfoundation devices
    result = subprocess.run(
        ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # The device information will be in stderr
    output = result.stderr

    video_devices = {}

    # Extract lines containing camera information
    is_video_section = False
    for line in output.split('\n'):
        # Look for the start of the video section
        if 'AVFoundation video devices' in line:
            is_video_section = True
            continue
        # End processing after the video section
        if 'AVFoundation audio devices' in line:
            break
        
        # Process only lines that are part of the video devices section
        if is_video_section and '[' in line and ']' in line:
            # Look for the second pair of brackets, which contains the index
            try:
                bracket_pairs = [i for i in range(len(line)) if line[i] == '[']
                if len(bracket_pairs) >= 2:  # Make sure there are at least two pairs of brackets
                    index = int(line[bracket_pairs[1] + 1:line.find(']', bracket_pairs[1])])
                    device_name = line[line.find(']') + 1:].strip()  # Extract the device name after the second bracket
                    video_devices[index] = device_name
            except ValueError:
                continue

    return video_devices



# Function to add transparent rectangle with multiline text at the bottom of the frame
def add_caption_to_frame(frame, text="Placeholder caption", font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=1):
    img_height, img_width, _ = frame.shape
    
    # Wrap text into multiple lines
    lines = wrap_text(text, font, font_scale, thickness, img_width)

    # Calculate the height required for the text block
    text_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1]
    line_spacing = 5  # Space between lines
    total_text_height = len(lines) * (text_height + line_spacing)

    # Coordinates for the rectangle background (adjust height based on number of lines)
    rect_x1, rect_y1 = 0, img_height - total_text_height - 30  # 30 is some padding from the bottom
    rect_x2, rect_y2 = img_width, img_height

    # Create a transparent overlay
    overlay = frame.copy()

    # Add a white rectangle with some transparency
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)

    # Blend the rectangle with the frame to create transparency (0.6 is the transparency level)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw each line of text
    y_offset = img_height - total_text_height - 15  # Start drawing above the rectangle
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_width = text_size[0]
        text_x = (img_width - text_width) // 2  # Center the text
        cv2.putText(frame, line, (text_x, y_offset), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y_offset += text_height + line_spacing  # Move to the next line
    
    return frame




# Function to continuously capture and display images from the selected camera with caption
def capture_and_display_continuous(camera_index, frame_rate):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open the camera at index {camera_index}.")
        return

    try:
        while True:
            # Capture a single frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read the frame.")
                break

            # Add the caption and the transparent background to the frame
            text = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor."
            frame_with_caption = add_caption_to_frame(frame, text=text+text+text)

            # Display the frame with caption using OpenCV
            cv2.imshow('Webcam Feed with Caption', frame_with_caption)

            # Check if the 'q' key is pressed to break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting the loop.")
                break
            
            # Sleep to control the frame rate (1 / frame_rate seconds between frames)
            #time.sleep(1 / frame_rate)
    finally:
        # Release the webcam and close all OpenCV windows when finished
        cap.release()
        cv2.destroyAllWindows()


# Command-line interface to select the camera by index and frame rate
def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Camera Selector for Webcam Stream with Frame Rate Control")
    parser.add_argument("-i", "--index", type=int, required=True, help="Index of the camera to use")
    parser.add_argument("-f", "--framerate", type=float, required=False, default=30.0, help="Frame rate to capture (Hz)")

    # Parse arguments
    args = parser.parse_args()
    camera_index = args.index
    frame_rate = args.framerate

    # List the available video devices (cameras)
    devices = list_available_cameras()

    # Print the available devices
    print("Available cameras:")
    for index, name in devices.items():
        print(f"[{index}] {name}")

    # Check if the provided index is valid
    if camera_index not in devices:
        print(f"Error: No camera found with index {camera_index}")
        return

    # Start capturing from the selected camera at the specified frame rate
    capture_and_display_continuous(camera_index, frame_rate)


# Start the script
if __name__ == "__main__":
    main()
