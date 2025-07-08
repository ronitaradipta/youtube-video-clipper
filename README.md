# AI Video Clipper

An intelligent video processing API that automatically transcribes videos, identifies engaging moments, and creates short clips with subtitles. Perfect for content creators who want to extract highlights from longer videos.

## Features

- 🎬 **Video Transcription**: Automatic speech-to-text using Azure OpenAI Whisper
- 🤖 **Smart Moment Detection**: AI-powered identification of interesting video segments
- ✂️ **Automatic Clipping**: Creates 30-60 second clips from identified moments
- 📱 **Vertical Video Conversion**: Converts horizontal videos to vertical format (1080x1920) with blurred background
- 📝 **Subtitle Generation**: Automatically adds subtitles to video clips
- 📺 **YouTube Integration**: Direct YouTube video download and processing
- 🔒 **Secure API**: Bearer token authentication

## Prerequisites

Before running the application, ensure you have:

- Python 3.8+ (recommended 3.12)
- FFmpeg installed and accessible in PATH
- Azure OpenAI API credentials
- Required Python packages (see requirements below)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-video-clipper
   ```

2. **Create and activate virtual environment (recommended)**

   - **Windows**:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   - **macOS**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**

   - **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

5. **Set up environment variables**
   Create a `.env` file in the root directory:

   ```env
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AUTH_TOKEN=your_secure_auth_token
   ```

6. **Create required directories**
   ```bash
   mkdir input_videos output_clips temp
   ```

## Usage

### Starting the Server

Run the FastAPI server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check

```http
GET /
```

**Response:**

```json
{
  "message": "AI Video Clipper API",
  "version": "1.0.0"
}
```

#### 2. Download YouTube Video

```http
POST /download-yt
```

**Headers:**

```
Authorization: Bearer <your_auth_token>
```

**Request Body:**

```json
{
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "full_video": false
}
```

**Parameters:**

- `url`: YouTube video URL
- `full_video`: `true` for full video, `false` to trim to 10 minutes

**Response:**

```json
{
  "success": true,
  "message": "Video downloaded and trimmed successfully",
  "filename": "input_videos/Video_Title_12345678.mp4"
}
```

#### 3. Transcribe Video

```http
POST /transcribe
```

**Headers:**

```
Authorization: Bearer <your_auth_token>
```

**Request Body:**

```json
{
  "video_path": "Video_Title_12345678.mp4"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Video transcribed successfully",
  "transcription": "[{\"start\": 0.0, \"end\": 5.2, \"word\": \"Hello everyone\"}, ...]"
}
```

#### 4. Process Video (Full Pipeline)

```http
POST /process
```

**Headers:**

```
Authorization: Bearer <your_auth_token>
```

**Request Body:**

```json
{
  "video_path": "Video_Title_12345678.mp4",
  "full_video": false
}
```

**Response:**

```json
{
  "success": true,
  "message": "Video processing completed successfully",
  "clips_created": 3,
  "run_id": "uuid-generated-run-id"
}
```

### Example Workflow

1. **Download a YouTube video:**

   ```bash
   curl -X POST "http://localhost:8000/download-yt" \
     -H "Authorization: Bearer your_auth_token" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://youtube.com/watch?v=dQw4w9WgXcQ", "full_video": false}'
   ```

2. **Process the video:**

   ```bash
   curl -X POST "http://localhost:8000/process" \
     -H "Authorization: Bearer your_auth_token" \
     -H "Content-Type: application/json" \
     -d '{"video_path": "Never_Gonna_Give_You_Up_12345678.mp4"}'
   ```

3. **Check output clips:**
   Processed clips will be saved in `output_clips/<run_id>/clip_0.mp4`, `clip_1.mp4`, etc.

## Configuration

### Environment Variables

| Variable                           | Description                         | Required |
| ---------------------------------- | ----------------------------------- | -------- |
| `AZURE_OPENAI_API_KEY`             | Azure OpenAI API key                | Yes      |
| `AZURE_OPENAI_ENDPOINT`            | Azure OpenAI endpoint URL           | Yes      |
| `AUTH_TOKEN`                       | Bearer token for API authentication | Yes      |
| `AZURE_OPENAI_API_VERSION`         | Azure OpenAI API version            | Yes      |
| `AZURE_OPENAI_TRANSCRIPTION_MODEL` | Azure OpenAI transcription model    | Yes      |
| `AZURE_OPENAI_COMPLETION_MODEL`    | Azure OpenAI chat completion model  | Yes      |

### Directory Structure

```
ai-video-clipper/
├── main.py
├── models.py
├── .env
├── input_videos/          # Downloaded/uploaded videos
├── output_clips/          # Generated clips
├── temp/                  # Temporary processing files
└── README.md
```

## Models Required

The application uses the following Azure OpenAI models:

- **Whisper**: `clloh-n8n-whisper` for audio transcription
- **GPT-4**: `clloh-n8n-gpt-4.1-mini` for moment identification

Make sure these models are deployed in your Azure OpenAI service.

## Video Processing Pipeline

1. **Video Input**: Upload video or download from YouTube
2. **Audio Extraction**: Extract audio track using FFmpeg
3. **Transcription**: Convert speech to text with timestamps
4. **Moment Detection**: AI identifies interesting 30-60 second segments
5. **Clip Creation**: Extract video segments and convert to vertical format
6. **Subtitle Generation**: Add subtitles with proper styling
7. **Output**: Save processed clips with subtitles

## Features in Detail

### Vertical Video Conversion

- Converts horizontal videos to 1080x1920 format
- Creates blurred background for horizontal content
- Maintains aspect ratio while fitting content

### Subtitle Generation

- Automatic word-level timing
- Customizable styling (font, size, color, outline)
- Bottom-center positioning optimized for mobile viewing
- Groups words into readable chunks (max 5 words per subtitle)

### Smart Moment Detection

The AI identifies clips based on:

- Complete, self-contained ideas
- Compelling stories or explanations
- Emotional moments or reactions
- Tutorial segments
- Entertainment value

## Troubleshooting

### Common Issues

1. **FFmpeg not found**

   - Ensure FFmpeg is installed and in your system PATH
   - Test with `ffmpeg -version`

2. **Azure OpenAI errors**

   - Check your API key and endpoint
   - Verify model deployments exist
   - Check API quotas and limits

3. **Video processing failures**

   - Ensure input video files exist in `input_videos/`
   - Check video format compatibility
   - Review FFmpeg error logs

4. **Subtitle rendering issues**
   - Verify subtitle file generation
   - Check FFmpeg subtitle filter support
   - Review font availability

### Debug Mode

Enable debug logging by adding print statements or using Python's logging module to trace the processing pipeline.

## Performance Considerations

- Processing time depends on video length and complexity
- Transcription is the most time-consuming step
- Clips are processed sequentially (can be parallelized for better performance)
- Temporary files are cleaned up after processing

## Security

- All API endpoints require bearer token authentication
- Temporary files are automatically cleaned up
- No persistent storage of sensitive data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:

- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review FFmpeg and Azure OpenAI documentation

---

**Note**: This application requires valid Azure OpenAI credentials and FFmpeg installation. Make sure to configure these properly before running the application.
