import json
import pathlib
import shutil
import subprocess
import time
import uuid
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import os
import re
from models import ProcessVideoRequest, ProcessVideoResponse, TranscribeVideoRequest, TranscribeVideoResponse, DownloadVideoRequest, DownloadVideoResponse
from openai import AzureOpenAI
from pytubefix import YouTube
from pytubefix.cli import on_progress

import pysubs2
from dotenv import load_dotenv

load_dotenv()

auth_scheme = HTTPBearer()
app = FastAPI()

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def create_vertical_video_simple(input_video_path: pathlib.Path, output_video_path: pathlib.Path):
    """
    Create a vertical video (1080x1920) from horizontal input video
    with blurred background and centered original video
    """
    # Ensure output directory exists
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # First, get video dimensions to calculate scaling
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", 
        "-show_streams", str(input_video_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        audio_stream = None
        for stream in video_info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_stream = stream
        
        if not video_stream:
            raise RuntimeError("No video stream found")
            
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        print(f"Original video dimensions: {width}x{height}")
        print(f"Audio stream found: {audio_stream is not None}")
        
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not probe video dimensions: {e}")
        # Use default filter without dimension checking
        width, height = 1920, 1080  # Assume typical horizontal video
        audio_stream = None
    
    # Calculate scaling for different scenarios
    target_width = 1080
    target_height = 1920
    
    # Method 1: Create blurred background that fills the vertical frame
    # Method 2: Scale original video to fit within the frame while maintaining aspect ratio
    
    if width > height:  # Horizontal video
        # For horizontal videos, we need to:
        # 1. Create a blurred background scaled to fill 1080x1920
        # 2. Scale the original video to fit within the frame
        # 3. Center the scaled video on the blurred background
        
        filter_complex = (
            # Create blurred background - scale to fill entire vertical frame
            f"[0:v]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height},"
            f"boxblur=luma_radius=20:luma_power=1[blurred];"
            
            # Scale original video to fit within frame while maintaining aspect ratio
            f"[0:v]scale={target_width}:-1:force_original_aspect_ratio=decrease[scaled];"
            
            # Overlay scaled video on blurred background (centered)
            f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2[final]"
        )
        
        # Improved audio handling
        if audio_stream:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video_path),
                "-filter_complex", filter_complex,
                "-map", "[final]",
                "-map", "0:a",  # Map audio stream
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",  # Use AAC instead of copy for better compatibility
                "-b:a", "128k",  # Set audio bitrate
                "-ar", "44100",  # Set audio sample rate
                str(output_video_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video_path),
                "-filter_complex", filter_complex,
                "-map", "[final]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",  # No audio
                str(output_video_path)
            ]
        
    else:  # Vertical video or square
        # For vertical videos, just scale to target dimensions
        filter_complex = (
            f"[0:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black[final]"
        )
        
        if audio_stream:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video_path),
                "-filter_complex", filter_complex,
                "-map", "[final]",
                "-map", "0:a",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                str(output_video_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video_path),
                "-filter_complex", filter_complex,
                "-map", "[final]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",
                str(output_video_path)
            ]
    
    print(f"Running FFmpeg command: {' '.join(cmd)}")
    
    try:
        # Run with error capture
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Vertical video created successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        # Try alternative simpler approach
        print("Trying alternative approach...")

        # Check if input has audio first
        has_audio_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a", 
            "-show_entries", "stream=index", "-of", "csv=p=0", 
            str(input_video_path)
        ]

        try:
            audio_check = subprocess.run(has_audio_cmd, capture_output=True, text=True, check=False)
            has_audio = bool(audio_check.stdout.strip())
            print(f"Input has audio: {has_audio}")
        except:
            has_audio = False

        if has_audio:
            simple_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video_path),
                "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                       f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                str(output_video_path)
            ]
        else:
            simple_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video_path),
                "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                       f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",
                str(output_video_path)
            ]
        
        try:
            subprocess.run(simple_cmd, check=True, capture_output=True, text=True)
            print("Vertical video created with simple approach")
        except subprocess.CalledProcessError as e2:
            print(f"Simple approach also failed:")
            print(f"STDERR: {e2.stderr}")
            raise RuntimeError(f"Both FFmpeg approaches failed. Last error: {e2.stderr}")


def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: pathlib.Path, output_path: pathlib.Path, base_dir: pathlib.Path, max_words: int = 5):
    """
    Create subtitles and overlay them on the video using FFmpeg
    Fixed version with proper path handling and error checking
    """

    # Convert paths to pathlib objects for consistent handling
    clip_video_path = pathlib.Path(clip_video_path)
    output_path = pathlib.Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    subtitle_path = pathlib.Path(base_dir) / "subtitles.ass"
    
    print(f"Creating subtitle file at: {subtitle_path}")
    print(f"Input video: {clip_video_path}")
    print(f"Output video: {output_path}")
    
    # Verify input video exists
    if not clip_video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {clip_video_path}")
    
    # Filter segments that fall within the clip timeframe
    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]
    
    print(f"Found {len(clip_segments)} segments for clip")
    
    # Group words into subtitle chunks
    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        # Convert to relative time (relative to clip start)
        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= 0:
            continue

        if not current_words:
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        elif len(current_words) >= max_words:
            subtitles.append(
                (current_start, current_end, ' '.join(current_words)))
            current_words = [word]
            current_start = start_rel
            current_end = end_rel
        else:
            current_words.append(word)
            current_end = end_rel

    if current_words:
        subtitles.append(
            (current_start, current_end, ' '.join(current_words)))

    print(f"Created {len(subtitles)} subtitle segments")

    # Create ASS subtitle file
    subs = pysubs2.SSAFile()

    # Set ASS file properties
    subs.info["WrapStyle"] = "0"
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    subs.info["ScriptType"] = "v4.00+"

    # Create subtitle style
    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Arial"  # Use Arial instead of Anton for better compatibility
    new_style.fontsize = 48  # Reduced font size for better fit
    new_style.primarycolor = pysubs2.Color(255, 255, 255)  # White text
    new_style.outline = 2.0
    new_style.outlinecolor = pysubs2.Color(0, 0, 0)  # Black outline
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)  # Semi-transparent black shadow
    new_style.alignment = 2  # Bottom center
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 200
    new_style.spacing = 0.0
    new_style.bold = True

    subs.styles[style_name] = new_style

    # Add subtitle events
    for i, (start, end, text) in enumerate(subtitles):
        # Clean text - remove extra whitespace and problematic characters
        clean_text = ' '.join(text.split())
        clean_text = clean_text.replace('\\', '\\\\').replace('"', '\\"')
        
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        
        line = pysubs2.SSAEvent(
            start=start_time, 
            end=end_time, 
            text=clean_text, 
            style=style_name
        )
        subs.events.append(line)

    # Save subtitle file
    try:
        subs.save(str(subtitle_path))
        print(f"Subtitle file saved: {subtitle_path}")
        
        # Verify subtitle file was created and has content
        if not subtitle_path.exists():
            raise FileNotFoundError("Subtitle file was not created")
            
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError("Subtitle file is empty")
                
        print(f"Subtitle file size: {subtitle_path.stat().st_size} bytes")
        
    except Exception as e:
        print(f"Error saving subtitle file: {e}")
        raise RuntimeError(f"Failed to create subtitle file: {e}")

    # Method 1: Try using subtitles filter (more reliable than ass filter)
    methods = [
        {
            "name": "subtitles_filter",
            "cmd": [
                "ffmpeg", "-y",
                "-i", str(clip_video_path),
                "-vf", f"subtitles={str(subtitle_path).replace(chr(92), '/')}",  # Convert backslashes to forward slashes
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "copy",  # Copy audio if present
                str(output_path)
            ]
        },
        {
            "name": "ass_filter",
            "cmd": [
                "ffmpeg", "-y",
                "-i", str(clip_video_path),
                "-vf", f"ass={str(subtitle_path).replace(chr(92), '/')}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "copy",
                str(output_path)
            ]
        },
        {
            "name": "no_subtitles",
            "cmd": [
                "ffmpeg", "-y",
                "-i", str(clip_video_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "copy",
                str(output_path)
            ]
        }
    ]

    # Try each method
    for i, method in enumerate(methods):
        print(f"Trying method {i+1}: {method['name']}")
        
        try:
            result = subprocess.run(
                method['cmd'], 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            print(f"Success with method: {method['name']}")
            
            # Verify output file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Output file created successfully: {output_path}")
                break
            else:
                print(f"Output file was not created properly")
                if i < len(methods) - 1:
                    continue
                else:
                    raise RuntimeError("Output file was not created")
                    
        except subprocess.TimeoutExpired:
            print(f"Method {method['name']} timed out")
            if i < len(methods) - 1:
                continue
            else:
                raise RuntimeError("All methods timed out")
                
        except subprocess.CalledProcessError as e:
            print(f"Method {method['name']} failed:")
            print(f"Return code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            
            if i < len(methods) - 1:
                print("Trying next method...")
                continue
            else:
                raise RuntimeError(f"All methods failed. Last error: {e.stderr}")

def process_clip(base_dir: pathlib.Path, original_video_path: pathlib.Path, start_time: float, end_time: float,
                 clip_index: int, transcript_segments: list, output_path: pathlib.Path):
    try:
        clip_name = f"clip_{clip_index}"
        clip_dir = base_dir / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
        vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
        subtitle_output_path = output_path
        
        pyavi_path = clip_dir / "pyavi"
        audio_path = pyavi_path / "audio.wav"

        pyavi_path.mkdir(parents=True, exist_ok=True)

        duration = end_time - start_time
        cut_command = (f'ffmpeg -y -i "{str(original_video_path)}" -ss {start_time} -t {duration} "{str(clip_segment_path)}"')
        subprocess.run(cut_command, shell=True, check=True, capture_output=True, text=True)

        extract_cmd = f'ffmpeg -y -i "{str(clip_segment_path)}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{str(audio_path)}"'
        subprocess.run(extract_cmd, shell=True, check=True, capture_output=True, text=True)

        shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

        cvv_start_time = time.time()
       
        create_vertical_video_simple(
            input_video_path=clip_segment_path,
            output_video_path=vertical_mp4_path
        )
        cvv_end_time = time.time()
        print(
            f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

        create_subtitles_with_ffmpeg(transcript_segments, start_time,
                                    end_time, vertical_mp4_path, subtitle_output_path, base_dir, max_words=5)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e.stderr}")
        raise RuntimeError(f"Failed running subprocess: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise RuntimeError(f"Failed processing clip {clip_index}: {e}")

class AiVideoClipper:
    def __init__(self):
        self.alignment_model = None
        self.metadata = None
        self.gemini_client = None
        self.model_loaded = False
        self.openai_client = None

    def load_model(self):
        if self.model_loaded:
            return
        
        print("Loading models")

        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        print("Models loaded...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}"'

        print("Extracting audio")
        subprocess.run(extract_cmd, shell=True,
                    check=True, capture_output=True)

        print("Extract audio finish")

        print("Starting transcription with Whisper...")
        start_time = time.time()

        with open(audio_path, "rb") as audio_file:
            result = self.openai_client.audio.transcriptions.create(
                file=audio_file,
                model=os.getenv("AZURE_OPENAI_TRANSCRIPTION_MODEL"),
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )

        duration = time.time() - start_time
        print("Transcription and alignment took " + str(duration) + " seconds")

        # Extract word-level segments
        segments = []
        
        if hasattr(result, 'words') and result.words:
            for item in result.words:
                segments.append({
                    "start": item.start,
                    "end": item.end,
                    "word": item.word,
                })
    
        # Fallback - use full text as single segment
        else:
            print("No detailed segments found, using full text")
            if hasattr(result, 'text'):
                segments.append({
                    "start": 0.0,
                    "end": duration,
                    "word": result.text,
                })

        print(f"Extracted {len(segments)} word segments")
        
        # Return segments as JSON string
        return json.dumps(segments)


    def identify_moments(self, transcript: dict):
        prompt = """
        You are given a transcript of spoken content from a video. Each transcript entry includes a sentence along with its start and end time in seconds.
        
        Your task is to identify and extract meaningful segments from this transcript that can be used to create short video clips.
        
        Please follow these guidelines:
        - Each clip should be between **30 and 60 seconds long**.
        - Ensure that clips do not overlap with one another.
        - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
        - Each clip should represent a **complete and self-contained idea**, such as:
            - a compelling story
            - a full explanation of a concept
            - a reaction or emotional moment
            - a demonstration or tutorial
            - a joke or entertaining bit
            - a question followed by an answer (if any)
        - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
        - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
        - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

        Avoid including:
        - Moments of greeting, thanking, or saying goodbye.
        - Non-question and answer interactions.

        If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

        The transcript is as follows:\n\n""" + str(transcript)

        response = self.openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_COMPLETION_MODEL"),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print(f"Identified moments response: {response.choices[0].message.content}")
        return response.choices[0].message.content

    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

        # Load models if not already loaded
        self.load_model()
        req_video_path = request.video_path

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("temp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load video file
            input_video_path = pathlib.Path("input_videos") / req_video_path
            if not input_video_path.exists():
                raise HTTPException(status_code=404, detail="Video file not found")
            # copy video to tmp
            shutil.copy(input_video_path, base_dir)
            video_path = base_dir / req_video_path

            # 1. Transcription
            transcript_segments_json = self.transcribe_video(base_dir, video_path)
            transcript_segments = json.loads(transcript_segments_json)

            # 2. Identify moments for clips
            print("Identifying clip moments")
            identified_moments_raw = self.identify_moments(transcript_segments)

            cleaned_json_string = identified_moments_raw.strip()
            if cleaned_json_string.startswith("```json"):
                cleaned_json_string = cleaned_json_string[len("```json"):].strip()
            if cleaned_json_string.endswith("```"):
                cleaned_json_string = cleaned_json_string[:-len("```")].strip()

            clip_moments = json.loads(cleaned_json_string)
            if not clip_moments or not isinstance(clip_moments, list):
                print("Error: Identified moments is not a list")
                clip_moments = []

            print(clip_moments)

            output_folder = pathlib.Path("output_clips")
            output_folder.mkdir(exist_ok=True)

            clips_created = 0
            # 3. Process clips
            for index, moment in enumerate(clip_moments[:5]):
                if "start" in moment and "end" in moment:
                    print("Processing clip" + str(index) + " from " +
                        str(moment["start"]) + " to " + str(moment["end"]))
                    process_clip(
                        base_dir=base_dir,
                        original_video_path=video_path,
                        clip_index=index,
                        start_time=moment["start"],
                        end_time=moment["end"],
                        transcript_segments=transcript_segments,
                        output_path=output_folder / run_id / f"clip_{index}.mp4"
                    )
                    clips_created += 1

            return ProcessVideoResponse(
                success=True,
                message="Video processing completed successfully",
                clips_created=clips_created,
                run_id=run_id
            )
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        
        finally:
            # cleanup
            if base_dir.exists():
                print(f"Cleaning up temp dir after {base_dir}")
                shutil.rmtree(base_dir, ignore_errors=True)

    def transcribe_video_only(self, request: TranscribeVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        if token.credentials != os.environ.get("AUTH_TOKEN", "123123"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})
        
        self.load_model()

        # Setup base_dir
        base_dir = pathlib.Path("input_videos")
        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            input_video_path = pathlib.Path("input_videos") / request.video_path

            print(f"video path : {request.video_path}")

            if not input_video_path.exists():
                raise HTTPException(status_code=404, detail="Video file not found")

            transcribed_video_json = self.transcribe_video(base_dir, input_video_path)

            return TranscribeVideoResponse(
                success=True,
                message="Video transcribed successfully",
                transcription=transcribed_video_json
            )
        except Exception as e:
            print(f"Error transcribing video: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error transcribing video: {str(e)}")

# Global instance
clipper = AiVideoClipper()

@app.get("/")
def read_root():
    return {"message": "AI Video Clipper API", "version": "1.0.0"}

@app.post("/transcribe", response_model=TranscribeVideoResponse)
def transcribe_video_endpoint(request: TranscribeVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    return clipper.transcribe_video_only(request, token)

@app.post("/process", response_model=ProcessVideoResponse)
def process_video_endpoint(request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    return clipper.process_video(request, token)

@app.post("/download-yt", response_model=DownloadVideoResponse)
def download_yt_endpoint(request:DownloadVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

    try:
        # Generate unique filename
        video_id = str(uuid.uuid4())[:8]
        
        yt = YouTube(request.url, on_progress_callback=on_progress)
        print(f"Downloading video : {yt.title}")

        stream = yt.streams.get_highest_resolution()

        title_safe = sanitize_filename(yt.title).replace(" ", "_")
        temp_output = f"temp_{video_id}.mp4"
        final_output = f"{title_safe}_{video_id}.mp4"
        output_path = pathlib.Path("input_videos") / final_output

        # Download temporary
        stream.download(filename=temp_output)

        if request.full_video:
            # save full video
            os.rename(temp_output, output_path)
        else:
            # trim max 10 minutes
            cut_command = [
                "ffmpeg", "-y",
                "-i", temp_output,
                "-t", "600",
                "-c", "copy",
                str(output_path)
            ]

            subprocess.run(cut_command, check=True, capture_output=True)

            # Delete file temporary
            os.remove(temp_output)

        return DownloadVideoResponse(
            success=True,
            message="Video downloaded and trimmed successfully",
            filename=str(output_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading video: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)