from pydantic import BaseModel
from typing import Optional

class ProcessVideoRequest(BaseModel):
    video_path: str

class ProcessVideoResponse(BaseModel):
    success: bool
    message: str
    clips_created: int
    run_id: str

class TranscribeVideoRequest(BaseModel):
    video_path: str

class TranscribeVideoResponse(BaseModel):
    success: bool
    message: str
    transcription: str

class DownloadVideoRequest(BaseModel):
    url: str
    full_video: Optional[bool] = False

class DownloadVideoResponse(BaseModel):
    success: bool
    message: str
    