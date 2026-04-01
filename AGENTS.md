# Voicebox - Voice Design Studio

Voice design studio for creating and customizing AI voices. 521 files.

## Project Structure

```
voicebox/
├── backend/         # Python TTS/STT service
├── app/            # React app (shadcn/ui)
├── web/            # Web client
├── tauri/          # Desktop app (Rust)
├── docs/           # Documentation
├── landing/        # Marketing landing page
└── Dockerfile
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| TTS/STT backends | `backend/backends/` | faster-whisper, Coqui, etc. |
| API routes | `backend/routes/` | FastAPI endpoints |
| Frontend components | `app/src/components/ui/` | shadcn/ui |
| Desktop app | `tauri/src-tauri/` | Rust/Tauri |
| Docker | `docker-compose.yml` | GPU-enabled (NVIDIA runtime) |

## Tech Stack

- **Backend**: Python with FastAPI, faster-whisper (STT), multiple TTS backends
- **Frontend**: React, TypeScript, Tailwind, shadcn/ui
- **Desktop**: Tauri (Rust)
- **GPU**: NVIDIA runtime enabled in docker-compose

## Anti-Patterns

- **faster-whisper integration**: Replaced whisper for 3-4x faster STT (backends/pytorch_backend.py)
- **Multiple entry points**: `backend/main.py`, `backend/server.py`, `backend/app.py`

## Development

```bash
# Run backend
cd voicebox/backend && python -m server

# Build Docker
cd voicebox && docker compose up --build

# Run frontend
cd voicebox/app && npm run dev
```