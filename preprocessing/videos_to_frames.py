import os
import subprocess

# Path to the root folder containing video folders
videos_root_dir = 'D:/BOLD_public/videos/003'

# Path to the folder where extracted frames will be saved
output_root_dir = 'F:/BOLD_public/mmextract/003'

# Ensure the output root folder exists
os.makedirs(output_root_dir, exist_ok=True)

# Loop through each folder in the root directory
for folder in os.listdir(videos_root_dir):
    folder_path = os.path.join(videos_root_dir, folder)

    if os.path.isdir(folder_path):
        # Create a corresponding folder for frames
        output_folder = os.path.join(output_root_dir, folder)
        os.makedirs(output_folder, exist_ok=True)

        # Loop through all videos in the current folder
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            video_path = video_path.replace('\\','/')
            
            if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Create a subfolder for the frames from this video
                video_name = os.path.splitext(video_file)[0]  # Get the video name without extension
                frame_output_folder = os.path.join(output_folder, video_name + '.mp4')
                os.makedirs(frame_output_folder, exist_ok=True)

                # FFmpeg command to extract frames
                ffmpeg_command = [
                r'C:\Users\Krist\Downloads\ffmpeg-7.1\ffmpeg-7.1\bin\ffmpeg.exe',
                '-i', video_path,
                #'-r', '2', # Set frame rate to 2 FPS
                os.path.join(frame_output_folder, 'frame_%04d.png')
            ]

                # Run FFmpeg command
                subprocess.run(ffmpeg_command)

                print(f"Extracted frames from {video_file} to {frame_output_folder}")