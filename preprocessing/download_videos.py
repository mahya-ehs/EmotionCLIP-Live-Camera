from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import json
import os
import subprocess
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import shutil
from functools import partial
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='youtube',
                    help='the dir to save the dataset')
parser.add_argument('--video_meta', type=str, default='youtube_playlist_tv.json',
                    help='the json file containing the URLs')
parser.add_argument('--source', type=str, default='youtube',
                    help='choose from [local, youtube]; "local": process videos from "save_dir"; "youtube": download video and transcript first')
parser.add_argument('--fps', type=int, default=24,
                    help='output fps for videos')
parser.add_argument('--n_workers', type=int, default=1,
                    help='number of workers')

args = parser.parse_args()


def get_youtube_video_with_transcript(ytid, save_dir='./youtube', transcript_only=False):
    # Process YouTube ID and link
    if 'https://www.youtube.com/watch' in ytid:
        ytid = ytid.split('?v=')[-1]
    link = f'https://www.youtube.com/watch?v={ytid}'
    print(link)
    save_path = os.path.join(save_dir, ytid)

    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transcript_path = f"{save_path}/transcript.json"

    # Get transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(ytid)
        with open(transcript_path, 'w') as outfile:
            json.dump(transcript, outfile)
    except Exception as e:
        print(f"Transcript error: {e}")
        return save_path, False

    # # If only transcript is needed, return early
    # if transcript_only:
    #     return save_path, True

    # Download video using yt-dlp
    try:
        ydl_opts = {
            'outtmpl': f'{save_path}/%(title)s.%(ext)s',  # Save video in the specific directory
            'format': 'bestvideo+bestaudio/best',         # Best video/audio quality
        }
        print(f"Starting yt-dlp download for {link}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        print(f"Downloaded video: {ytid}")
    except Exception as e:
        print(f"yt-dlp download error: {e}")
        return save_path, False

    return save_path, True


def filter_frames_by_subs(subs_file, fps=24):
    with open(subs_file, 'r') as outfile:
        transcript = json.load(outfile)
    transcript = list(filter(lambda sub: '[' not in sub['text'], transcript))
    useful_frames = np.concatenate([list(range(int(sub_range['start']*fps),
                                              int(sub_range['start']*fps + sub_range['duration']*fps)))
                                    for sub_range in transcript])
    return np.unique(useful_frames)  # cc can overlap


def remove_unuseful_frames(useful_frames, frames_dir):
    for frame_path in os.listdir(frames_dir):
        if int(frame_path.split('.')[0]) not in useful_frames:
            os.remove(os.path.join(frames_dir, frame_path))
    return True


def video_to_frames(ytid, save_dir='./youtube', fps=24):
    if 'https://www.youtube.com/watch' in ytid:
        ytid = ytid.split('?v=')[-1]
    save_path = os.path.join(save_dir, ytid)
    video_file = os.path.join(save_path, list(filter(lambda x: '.mp4' in x, os.listdir(save_path)))[0])
    frames_dir = os.path.join(save_path, 'frames')
    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
    command = ['ffmpeg', '-i', video_file, '-vf', f'fps={fps}', '-q:v', '5', f'{frames_dir}/%08d.jpg']
    status = subprocess.call(command)
    return save_path, frames_dir, video_file, status


def meta_to_frames(meta, save_dir='./youtube', fps=12, transcript_only=False, download=True):
    if download:
        save_path, success = get_youtube_video_with_transcript(meta['link'], save_dir=save_dir, transcript_only=transcript_only)
        if transcript_only:
            return meta, success
    else:
        success = True

    if success:
        save_path, frames_dir, video_file, status = video_to_frames(meta['link'], fps=fps)
        if status != 0:
            print('video to frames status:', status)
            return meta, False
    return meta, success


def update_progress_bar(_):
    progress_bar.update()


def work(a, b):
    print(a, b)


if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.source == 'youtube':
        with open(args.video_meta, 'r') as outfile:
            playlist_movies = json.load(outfile)
    else:
        playlist_movies = [{'link': link} for link in os.listdir(args.save_dir)]

    print(len(playlist_movies))
    global progress_bar
    progress_bar = tqdm(total=len(playlist_movies))
    pool = mp.Pool(args.n_workers)
    for meta in playlist_movies:
        time.sleep(3)
        pool.apply_async(partial(meta_to_frames, save_dir=args.save_dir, fps=args.fps, transcript_only=True, download=args.source == 'youtube'),
                         (meta,), callback=update_progress_bar)

    pool.close()
    pool.join()
