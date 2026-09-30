# ClipSum: AI Video Assistant ( https://clipsum-aivideoagent.streamlit.app )

Upload a video or audio file, or paste a link, and get a summary, action items, key decisions and open questions. Then ask follow-up questions about anything that was said in the video.

**Live app:** add your Streamlit link here

## Features

- Speech-to-text with OpenAI Whisper (runs locally, no transcription API needed)
- Title, bullet-point summary, action items, key decisions and open questions, generated with Mistral
- Chat with the transcript using retrieval-augmented generation (ChromaDB + sentence-transformer embeddings)
- Download the transcript and the report

## Using the hosted app

The hosted version works best with **uploaded files** and with **direct links to public, freely available media** (for example a link ending in `.mp3` or `.mp4` from an open host such as archive.org).

**YouTube links do not work on the hosted app**, and links from some other platforms (Reddit, Instagram, TikTok and similar) may fail too.

### Why

The app downloads videos with `yt-dlp`. On a personal computer, the request comes from a home internet connection, and YouTube allows it. On a hosted app, the request comes from a cloud data-center IP address. YouTube and several other platforms reject requests from these addresses on purpose, to stop automated downloading, and the download fails with `HTTP 403` before the app gets any audio. This is a restriction of the hosting environment, not a bug in the app.

Working around it would mean using logged-in browser cookies or a paid proxy. Both break often, and they conflict with the platforms' terms of use, so this project doesn't do that.

### What to do instead

| Situation | Option |
|---|---|
| You have the video file | Use **Upload a file** (most reliable) |
| The video is public and freely available | Paste a direct file link |
| You want to analyze a YouTube video | Run the app locally (below) |

## Run locally (YouTube links work here)

Requirements: Python 3.10+ and [ffmpeg](https://ffmpeg.org/) installed.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file next to `app.py`:

```dotenv
MISTRAL_API_KEY=your_key_here
```

Start the app:

```bash
streamlit run app.py
```

The first run is slower because Whisper and the embedding model download once.

## Deploying

Add `MISTRAL_API_KEY` in your host's secrets settings. The `.env` file is not deployed because it is gitignored. On Streamlit Community Cloud, `packages.txt` installs `ffmpeg`.

## Limits

- Mistral's free tier has rate limits, so keep test videos short (a few minutes) at first.
- Whisper runs on the server's CPU, so long videos take a while to transcribe.

## Stack

Streamlit, LangChain, Mistral AI, OpenAI Whisper, ChromaDB, sentence-transformers, yt-dlp, pydub
