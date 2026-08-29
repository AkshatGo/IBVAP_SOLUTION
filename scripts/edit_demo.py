"""
IBVAP Video Editor
Post-processing utilities for demo videos
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import os


class VideoEditor:
    """
    Video editing utilities for demo post-processing
    """
    
    def __init__(self, input_dir: str = "demo_videos", output_dir: str = "final_videos"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def concat_videos(self, video_files: List[str], output_file: str) -> str:
        """
        Concatenate multiple videos into one
        
        Args:
            video_files: List of video file paths
            output_file: Output video filename
            
        Returns:
            Path to output video
        """
        if not video_files:
            print("[VideoEditor] No videos to concatenate")
            return ""
        
        # Get video properties from first file
        cap = cv2.VideoCapture(str(self.input_dir / video_files[0]))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create output writer
        output_path = self.output_dir / output_file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Concatenate all videos
        for video_file in video_files:
            video_path = self.input_dir / video_file
            if not video_path.exists():
                print(f"[VideoEditor] Warning: {video_file} not found, skipping")
                continue
            
            cap = cv2.VideoCapture(str(video_path))
            print(f"[VideoEditor] Adding: {video_file}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize if needed
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                writer.write(frame)
            
            cap.release()
        
        writer.release()
        print(f"[VideoEditor] Concatenated video saved: {output_path}")
        return str(output_path)
    
    def add_title_card(
        self,
        video_file: str,
        title: str,
        duration: float = 3.0,
        output_file: str = None
    ) -> str:
        """
        Add a title card at the beginning of a video
        
        Args:
            video_file: Input video filename
            title: Title text to display
            duration: Duration of title card in seconds
            output_file: Output video filename
            
        Returns:
            Path to output video
        """
        if output_file is None:
            output_file = f"titled_{video_file}"
        
        # Get video properties
        cap = cv2.VideoCapture(str(self.input_dir / video_file))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create output writer
        output_path = self.output_dir / output_file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Generate title card frames
        title_frames = int(fps * duration)
        for _ in range(title_frames):
            frame = self._create_title_frame(title, width, height)
            writer.write(frame)
        
        # Add original video
        cap = cv2.VideoCapture(str(self.input_dir / video_file))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        
        writer.release()
        print(f"[VideoEditor] Title card added: {output_path}")
        return str(output_path)
    
    def _create_title_frame(self, title: str, width: int, height: int) -> np.ndarray:
        """Create a title frame"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(height):
            gradient = int(13 + (y / height) * 20)
            frame[y, :] = [gradient, gradient + 2, gradient + 5]
        
        # Title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5
        thickness = 4
        
        (text_width, text_height), baseline = cv2.getTextSize(
            title, font, font_scale, thickness
        )
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw shadow
        cv2.putText(frame, title, (x + 3, y + 3), font, font_scale, (0, 0, 0), thickness + 2)
        # Draw title
        cv2.putText(frame, title, (x, y), font, font_scale, (88, 166, 255), thickness)
        
        return frame
    
    def add_transition(
        self,
        video1_file: str,
        video2_file: str,
        transition_frames: int = 30,
        output_file: str = None
    ) -> str:
        """
        Add a crossfade transition between two videos
        
        Args:
            video1_file: First video filename
            video2_file: Second video filename
            transition_frames: Number of frames for transition
            output_file: Output video filename
            
        Returns:
            Path to output video
        """
        if output_file is None:
            output_file = f"transition_{video1_file}"
        
        # Open videos
        cap1 = cv2.VideoCapture(str(self.input_dir / video1_file))
        cap2 = cv2.VideoCapture(str(self.input_dir / video2_file))
        
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        
        # Create output writer
        output_path = self.output_dir / output_file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Read all frames from video1
        frames1 = []
        while cap1.isOpened():
            ret, frame = cap1.read()
            if not ret:
                break
            frames1.append(frame)
        cap1.release()
        
        # Read all frames from video2
        frames2 = []
        while cap2.isOpened():
            ret, frame = cap2.read()
            if not ret:
                break
            frames2.append(frame)
        cap2.release()
        
        # Write video1 frames (except last transition_frames)
        for frame in frames1[:-transition_frames]:
            writer.write(frame)
        
        # Write transition frames (crossfade)
        for i in range(transition_frames):
            alpha = i / transition_frames
            frame1 = frames1[-(transition_frames - i)]
            frame2 = frames2[i]
            
            # Ensure frames are same size
            if frame1.shape != frame2.shape:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
            blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            writer.write(blended)
        
        # Write remaining video2 frames
        for frame in frames2[transition_frames:]:
            writer.write(frame)
        
        writer.release()
        print(f"[VideoEditor] Transition added: {output_path}")
        return str(output_path)
    
    def add_text_overlay(
        self,
        video_file: str,
        text: str,
        position: Tuple[int, int] = (50, 50),
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        output_file: str = None
    ) -> str:
        """
        Add text overlay to a video
        
        Args:
            video_file: Input video filename
            text: Text to overlay
            position: (x, y) position for text
            font_scale: Font size scale
            color: Text color (B, G, R)
            output_file: Output video filename
            
        Returns:
            Path to output video
        """
        if output_file is None:
            output_file = f"overlay_{video_file}"
        
        cap = cv2.VideoCapture(str(self.input_dir / video_file))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_path = self.output_dir / output_file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add text with shadow
            cv2.putText(frame, text, (position[0] + 2, position[1] + 2),
                       font, font_scale, (0, 0, 0), 3)
            cv2.putText(frame, text, position, font, font_scale, color, 2)
            
            writer.write(frame)
        
        cap.release()
        writer.release()
        print(f"[VideoEditor] Text overlay added: {output_path}")
        return str(output_path)
    
    def create_compilation(self, video_files: List[str], output_file: str = "compilation.mp4") -> str:
        """
        Create a compilation video with all videos concatenated
        
        Args:
            video_files: List of video filenames
            output_file: Output video filename
            
        Returns:
            Path to output video
        """
        # Add title card
        print("[VideoEditor] Creating compilation...")
        
        # Concatenate all videos
        return self.concat_videos(video_files, output_file)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IBVAP Video Editor")
    parser.add_argument("--input", default="demo_videos", help="Input directory")
    parser.add_argument("--output", default="final_videos", help="Output directory")
    parser.add_argument("--action", choices=["concat", "title", "transition", "overlay", "compile"],
                       required=True, help="Action to perform")
    parser.add_argument("--videos", nargs="+", help="Video files to process")
    parser.add_argument("--title", help="Title text for title card")
    parser.add_argument("--text", help="Text for overlay")
    parser.add_argument("--output-file", help="Output filename")
    
    args = parser.parse_args()
    
    editor = VideoEditor(input_dir=args.input, output_dir=args.output)
    
    if args.action == "concat" and args.videos:
        editor.concat_videos(args.videos, args.output_file or "concatenated.mp4")
    
    elif args.action == "title" and args.videos and args.title:
        editor.add_title_card(args.videos[0], args.title, output_file=args.output_file)
    
    elif args.action == "overlay" and args.videos and args.text:
        editor.add_text_overlay(args.videos[0], args.text, output_file=args.output_file)
    
    elif args.action == "compile" and args.videos:
        editor.create_compilation(args.videos, args.output_file)
    
    else:
        print("Invalid arguments. Use --help for usage information.")


if __name__ == "__main__":
    main()
