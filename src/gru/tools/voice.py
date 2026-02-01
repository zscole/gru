"""Voice message tools for Telegram integration.

Handles sending and receiving voice messages, including speech-to-text (STT)
and text-to-speech (TTS) conversion.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

import anthropic

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.config import Config
    from gru.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# Global references (set during initialization)
_config: Config | None = None
_orchestrator: Orchestrator | None = None


def set_voice_dependencies(config: Config, orchestrator: Orchestrator) -> None:
    """Set the config and orchestrator for voice tools."""
    global _config, _orchestrator
    _config = config
    _orchestrator = orchestrator


def get_config() -> Config | None:
    """Get the config instance."""
    return _config


def get_orchestrator() -> Orchestrator | None:
    """Get the orchestrator instance."""
    return _orchestrator


class VoiceConfig:
    """Configuration for voice message settings."""
    
    def __init__(self) -> None:
        # TTS settings
        self.tts_provider = os.getenv("GRU_TTS_PROVIDER", "eleven_labs")  # eleven_labs, openai, edge
        self.tts_voice_id = os.getenv("GRU_TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # ElevenLabs default
        self.tts_quality = os.getenv("GRU_TTS_QUALITY", "high")  # high, medium, low
        self.tts_speed = float(os.getenv("GRU_TTS_SPEED", "1.0"))  # 0.5 to 2.0
        
        # STT settings  
        self.stt_provider = os.getenv("GRU_STT_PROVIDER", "openai")  # openai, whisper (claude not supported)
        self.stt_language = os.getenv("GRU_STT_LANGUAGE", "en")  # language code
        
        # Audio format settings
        self.output_format = os.getenv("GRU_VOICE_FORMAT", "ogg")  # ogg, mp3, wav
        self.quality = os.getenv("GRU_VOICE_QUALITY", "medium")  # low, medium, high
        self.max_duration = int(os.getenv("GRU_VOICE_MAX_DURATION", "120"))  # seconds
        
        # API keys
        self.eleven_labs_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")


# Global config instance
_voice_config = VoiceConfig()


async def send_voice_message(
    chat_id: str, 
    text: str, 
    voice_id: str | None = None,
    speed: float | None = None,
    provider: str | None = None
) -> dict[str, Any]:
    """Send a voice message to a Telegram chat.
    
    Args:
        chat_id: Telegram chat ID to send to
        text: Text to convert to speech
        voice_id: Voice ID for TTS (provider-specific)
        speed: Speech speed (0.5 to 2.0)
        provider: TTS provider to use
        
    Returns:
        Dictionary with status and details
    """
    if not _orchestrator:
        return {"error": "Orchestrator not available"}
        
    try:
        provider = provider or _voice_config.tts_provider
        voice_id = voice_id or _voice_config.tts_voice_id
        speed = speed or _voice_config.tts_speed
        
        # Validate text length
        if len(text) > 4000:  # Most TTS services have limits
            return {"error": f"Text too long ({len(text)} chars). Maximum 4000 characters."}
            
        if not text.strip():
            return {"error": "Text cannot be empty"}
            
        # Generate audio based on provider
        audio_data = await _generate_speech(text, provider, voice_id, speed)
        if not audio_data:
            return {"error": f"Failed to generate speech with provider: {provider}"}
            
        # Send via orchestrator notification system
        # This will route through the appropriate bot interface
        try:
            chat_id_int = int(chat_id)
            # Send to specific user
            await _send_audio_to_telegram(chat_id_int, audio_data)
            return {"status": "sent", "chat_id": chat_id, "provider": provider, "length": len(text)}
        except ValueError:
            return {"error": f"Invalid chat_id: {chat_id}. Must be a numeric Telegram chat ID."}
            
    except Exception as e:
        logger.error(f"Failed to send voice message: {e}")
        return {"error": str(e)}


async def transcribe_voice_message(
    audio_data: str | bytes,
    format: str = "ogg",
    provider: str | None = None,
    language: str | None = None
) -> dict[str, Any]:
    """Transcribe audio data to text.
    
    Args:
        audio_data: Base64 encoded audio data or raw bytes
        format: Audio format (ogg, mp3, wav)
        provider: STT provider to use
        language: Language code for transcription
        
    Returns:
        Dictionary with transcription and metadata
    """
    try:
        provider = provider or _voice_config.stt_provider
        language = language or _voice_config.stt_language
        
        # Handle base64 data
        if isinstance(audio_data, str):
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception:
                return {"error": "Invalid base64 audio data"}
        else:
            audio_bytes = audio_data
            
        if len(audio_bytes) == 0:
            return {"error": "Empty audio data"}
            
        # Transcribe based on provider
        transcription = await _transcribe_audio(audio_bytes, format, provider, language)
        
        if not transcription:
            return {"error": f"Failed to transcribe with provider: {provider}"}
            
        return {
            "transcription": transcription,
            "provider": provider,
            "language": language,
            "audio_size": len(audio_bytes)
        }
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        return {"error": str(e)}


async def get_voice_settings() -> dict[str, Any]:
    """Get current voice message configuration."""
    return {
        "tts": {
            "provider": _voice_config.tts_provider,
            "voice_id": _voice_config.tts_voice_id,
            "quality": _voice_config.tts_quality,
            "speed": _voice_config.tts_speed,
            "available_providers": ["eleven_labs", "openai", "edge"],
        },
        "stt": {
            "provider": _voice_config.stt_provider,
            "language": _voice_config.stt_language,
            "available_providers": ["openai", "whisper"],
        },
        "audio": {
            "format": _voice_config.output_format,
            "quality": _voice_config.quality,
            "max_duration": _voice_config.max_duration,
            "supported_formats": ["ogg", "mp3", "wav"],
        },
        "api_keys_configured": {
            "eleven_labs": bool(_voice_config.eleven_labs_key),
            "openai": bool(_voice_config.openai_key),
        }
    }


async def update_voice_settings(
    tts_provider: str | None = None,
    tts_voice_id: str | None = None,
    tts_speed: float | None = None,
    stt_provider: str | None = None,
    stt_language: str | None = None,
    output_format: str | None = None
) -> dict[str, Any]:
    """Update voice message settings.
    
    Args:
        tts_provider: Text-to-speech provider
        tts_voice_id: Voice ID for TTS
        tts_speed: Speech speed (0.5 to 2.0)
        stt_provider: Speech-to-text provider
        stt_language: Language code for STT
        output_format: Audio output format
        
    Returns:
        Dictionary with updated settings
    """
    try:
        if tts_provider:
            if tts_provider not in ["eleven_labs", "openai", "edge"]:
                return {"error": f"Invalid TTS provider: {tts_provider}"}
            _voice_config.tts_provider = tts_provider
            
        if tts_voice_id:
            _voice_config.tts_voice_id = tts_voice_id
            
        if tts_speed is not None:
            if not 0.5 <= tts_speed <= 2.0:
                return {"error": "TTS speed must be between 0.5 and 2.0"}
            _voice_config.tts_speed = tts_speed
            
        if stt_provider:
            if stt_provider not in ["openai", "whisper"]:
                # Special message for claude since it was previously in config
                if stt_provider.lower() == "claude":
                    return {
                        "error": "Claude STT provider is not supported. Claude's API only supports PDF documents, not audio. "
                                "Use 'openai' for OpenAI Whisper API or 'whisper' for local Whisper."
                    }
                return {"error": f"Invalid STT provider: {stt_provider}. Supported: openai, whisper"}
            _voice_config.stt_provider = stt_provider
            
        if stt_language:
            _voice_config.stt_language = stt_language
            
        if output_format:
            if output_format not in ["ogg", "mp3", "wav"]:
                return {"error": f"Invalid format: {output_format}"}
            _voice_config.output_format = output_format
            
        return {
            "status": "updated",
            "settings": await get_voice_settings()
        }
        
    except Exception as e:
        logger.error(f"Failed to update voice settings: {e}")
        return {"error": str(e)}


# Internal helper functions

async def _generate_speech(text: str, provider: str, voice_id: str, speed: float) -> bytes | None:
    """Generate speech audio from text."""
    try:
        if provider == "eleven_labs":
            return await _generate_elevenlabs_speech(text, voice_id, speed)
        elif provider == "openai":
            return await _generate_openai_speech(text, voice_id, speed)
        elif provider == "edge":
            return await _generate_edge_speech(text, voice_id, speed)
        else:
            logger.error(f"Unknown TTS provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"Speech generation failed with {provider}: {e}")
        return None


async def _generate_elevenlabs_speech(text: str, voice_id: str, speed: float) -> bytes | None:
    """Generate speech using ElevenLabs API."""
    if not _voice_config.eleven_labs_key:
        logger.warning("ElevenLabs API key not configured")
        return None
        
    try:
        import httpx
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": _voice_config.eleven_labs_key
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.5,
                "use_speaker_boost": True,
                "speaking_rate": speed
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content
            
    except Exception as e:
        logger.error(f"ElevenLabs TTS failed: {e}")
        return None


async def _generate_openai_speech(text: str, voice: str, speed: float) -> bytes | None:
    """Generate speech using OpenAI TTS API."""
    if not _voice_config.openai_key:
        logger.warning("OpenAI API key not configured")
        return None
        
    try:
        import httpx
        
        # Map voice_id to OpenAI voice names
        voice_map = {
            "alloy": "alloy",
            "echo": "echo", 
            "fable": "fable",
            "onyx": "onyx",
            "nova": "nova",
            "shimmer": "shimmer"
        }
        openai_voice = voice_map.get(voice, "alloy")
        
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {_voice_config.openai_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": openai_voice,
            "speed": speed
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content
            
    except Exception as e:
        logger.error(f"OpenAI TTS failed: {e}")
        return None


async def _generate_edge_speech(text: str, voice: str, speed: float) -> bytes | None:
    """Generate speech using Edge TTS (free, no API key needed)."""
    try:
        import edge_tts
        
        # Edge TTS voice names
        edge_voice = voice if voice.startswith("en-") else "en-US-AriaNeural"
        rate = f"{int((speed - 1) * 100):+d}%"
        
        communicate = edge_tts.Communicate(text, edge_voice, rate=rate)
        
        # Collect audio data
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                
        return audio_data if audio_data else None
        
    except ImportError:
        logger.warning("edge-tts package not installed. Install with: pip install edge-tts")
        return None
    except Exception as e:
        logger.error(f"Edge TTS failed: {e}")
        return None


async def _transcribe_audio(audio_bytes: bytes, format: str, provider: str, language: str) -> str | None:
    """Transcribe audio bytes to text."""
    try:
        # Validate that we have audio data
        if not audio_bytes or len(audio_bytes) == 0:
            logger.error("Empty audio data provided for transcription")
            return None
            
        # Warn about Claude usage and redirect to supported providers
        if provider == "claude":
            logger.warning(
                f"Claude STT provider requested but not supported for audio transcription. "
                f"Claude's API only supports PDF documents, not audio files. "
                f"Falling back to supported provider."
            )
            return await _transcribe_with_claude(audio_bytes, format)
        elif provider == "openai":
            return await _transcribe_with_openai(audio_bytes, format)
        elif provider == "whisper":
            return await _transcribe_with_whisper(audio_bytes, format)
        else:
            logger.error(f"Unknown STT provider: {provider}. Supported: openai, whisper")
            # Fall back to OpenAI if available
            if _voice_config.openai_key:
                logger.info("Falling back to OpenAI Whisper API")
                return await _transcribe_with_openai(audio_bytes, format)
            else:
                logger.info("Falling back to local Whisper")
                return await _transcribe_with_whisper(audio_bytes, format)
    except Exception as e:
        logger.error(f"Transcription failed with {provider}: {e}")
        return None


async def _transcribe_with_claude(audio_bytes: bytes, format: str) -> str | None:
    """Transcribe using Claude's audio capabilities.
    
    Note: Claude API currently doesn't support direct audio transcription.
    Claude's document API only supports PDF files, not audio files.
    This function always falls back to supported transcription providers.
    """
    try:
        # IMPORTANT: Claude API does not support audio transcription
        # Attempting to send audio data as a document with PDF media type will fail
        # with: "messages.0.content.1.document.source.base64.media_type: Input should be 'application/pdf'"
        
        logger.info("Claude doesn't support audio transcription, using OpenAI Whisper instead")
        
        # Check if OpenAI is available before attempting fallback
        if not _voice_config.openai_key:
            logger.warning("OpenAI API key not configured, trying local Whisper")
            return await _transcribe_with_whisper(audio_bytes, format)
            
        return await _transcribe_with_openai(audio_bytes, format)
        
    except Exception as e:
        logger.error(f"Claude transcription fallback to OpenAI failed: {e}")
        # Try local Whisper as final fallback
        try:
            return await _transcribe_with_whisper(audio_bytes, format)
        except Exception as e2:
            logger.error(f"Local Whisper fallback also failed: {e2}")
            return None


async def _transcribe_with_openai(audio_bytes: bytes, format: str) -> str | None:
    """Transcribe using OpenAI Whisper API."""
    if not _voice_config.openai_key:
        logger.warning("OpenAI API key not configured")
        return None
        
    try:
        import httpx
        
        # Save to temporary file for upload
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {_voice_config.openai_key}"}
            
            with open(tmp_path, "rb") as audio_file:
                files = {"file": (f"audio.{format}", audio_file, f"audio/{format}")}
                data = {"model": "whisper-1"}
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, headers=headers, files=files, data=data, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    return result.get("text", "").strip()
                    
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
                
    except Exception as e:
        logger.error(f"OpenAI transcription failed: {e}")
        return None


async def _transcribe_with_whisper(audio_bytes: bytes, format: str) -> str | None:
    """Transcribe using local Whisper model."""
    try:
        import whisper
        import tempfile
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            return result["text"].strip()
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
                
    except ImportError:
        logger.warning("whisper package not installed. Install with: pip install openai-whisper")
        return None
    except Exception as e:
        logger.error(f"Local Whisper transcription failed: {e}")
        return None


async def _send_audio_to_telegram(chat_id: int, audio_data: bytes) -> None:
    """Send audio data to Telegram chat."""
    if not _orchestrator or not hasattr(_orchestrator, "telegram_bot"):
        logger.error("Telegram bot not available")
        return
        
    try:
        telegram_bot = getattr(_orchestrator, "telegram_bot", None)
        if not telegram_bot or not telegram_bot._app:
            logger.error("Telegram bot not initialized")
            return
            
        # Convert audio to voice message format
        audio_file = io.BytesIO(audio_data)
        audio_file.name = f"voice_message.{_voice_config.output_format}"
        
        await telegram_bot._app.bot.send_voice(
            chat_id=chat_id,
            voice=audio_file
        )
        
    except Exception as e:
        logger.error(f"Failed to send audio to Telegram: {e}")
        raise


async def get_voice_duration(audio_data: str | bytes, format: str = "ogg") -> dict[str, Any]:
    """Get the duration of an audio file in seconds.
    
    Args:
        audio_data: Base64 encoded audio data or raw bytes
        format: Audio format (ogg, mp3, wav, m4a)
        
    Returns:
        Dictionary with duration and metadata
    """
    try:
        # Handle base64 data
        if isinstance(audio_data, str):
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception:
                return {"error": "Invalid base64 audio data"}
        else:
            audio_bytes = audio_data
            
        if len(audio_bytes) == 0:
            return {"error": "Empty audio data"}
            
        # Save to temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            # Try to get duration using ffprobe (if available)
            try:
                import subprocess
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    duration = float(result.stdout.strip())
                    return {
                        "duration": duration,
                        "format": format,
                        "size": len(audio_bytes),
                        "method": "ffprobe"
                    }
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
                
            # Fallback: try pydub (if available)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(tmp_path, format=format)
                duration = len(audio) / 1000.0  # pydub returns milliseconds
                return {
                    "duration": duration,
                    "format": format,
                    "size": len(audio_bytes),
                    "method": "pydub"
                }
            except ImportError:
                logger.warning("pydub not available for audio duration calculation")
            except Exception as e:
                logger.warning(f"pydub failed to get duration: {e}")
                
            # Final fallback: estimate based on file size and format
            # This is very rough and not accurate, but better than nothing
            if format.lower() in ["ogg", "mp3"]:
                # Rough estimate: assume ~64kbps bitrate
                estimated_duration = len(audio_bytes) / (64 * 1000 / 8)  # 64 kbps in bytes/second
            else:
                # For uncompressed formats like WAV, use different estimation
                estimated_duration = len(audio_bytes) / (16000 * 2)  # Assume 16kHz mono 16-bit
                
            return {
                "duration": estimated_duration,
                "format": format,
                "size": len(audio_bytes),
                "method": "estimated",
                "warning": "Duration is estimated and may be inaccurate. Install ffmpeg or pydub for accurate results."
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
                
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        return {"error": str(e)}


def register_voice_tools() -> None:
    """Register all voice message tools."""
    
    register_tool(
        name="send_voice_message",
        description="Send a text message as a voice message to a Telegram chat. Converts text to speech and sends as audio.",
        parameters={
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID to send the voice message to (numeric string)",
            },
            "text": {
                "type": "string", 
                "description": "Text to convert to speech (max 4000 characters)",
            },
            "voice_id": {
                "type": "string",
                "description": "Voice ID to use for speech generation (provider-specific)",
                "optional": True,
            },
            "speed": {
                "type": "number",
                "description": "Speech speed multiplier (0.5 to 2.0, default 1.0)",
                "optional": True,
            },
            "provider": {
                "type": "string",
                "description": "TTS provider to use",
                "enum": ["eleven_labs", "openai", "edge"],
                "optional": True,
            },
        },
        handler=send_voice_message,
    )
    
    register_tool(
        name="transcribe_voice_message",
        description="Transcribe audio data to text using speech-to-text services.",
        parameters={
            "audio_data": {
                "type": "string",
                "description": "Base64 encoded audio data",
            },
            "format": {
                "type": "string",
                "description": "Audio format",
                "enum": ["ogg", "mp3", "wav", "m4a"],
                "optional": True,
            },
            "provider": {
                "type": "string", 
                "description": "STT provider to use",
                "enum": ["openai", "whisper"],
                "optional": True,
            },
            "language": {
                "type": "string",
                "description": "Language code for transcription (e.g., 'en', 'es', 'fr')",
                "optional": True,
            },
        },
        handler=transcribe_voice_message,
    )
    
    register_tool(
        name="get_voice_settings",
        description="Get current voice message configuration and available options.",
        parameters={},
        handler=get_voice_settings,
    )
    
    register_tool(
        name="update_voice_settings",
        description="Update voice message configuration settings.",
        parameters={
            "tts_provider": {
                "type": "string",
                "description": "Text-to-speech provider",
                "enum": ["eleven_labs", "openai", "edge"],
                "optional": True,
            },
            "tts_voice_id": {
                "type": "string", 
                "description": "Voice ID for TTS (provider-specific)",
                "optional": True,
            },
            "tts_speed": {
                "type": "number",
                "description": "Default speech speed (0.5 to 2.0)",
                "optional": True,
            },
            "stt_provider": {
                "type": "string",
                "description": "Speech-to-text provider", 
                "enum": ["openai", "whisper"],
                "optional": True,
            },
            "stt_language": {
                "type": "string",
                "description": "Language code for STT",
                "optional": True,
            },
            "output_format": {
                "type": "string",
                "description": "Audio output format",
                "enum": ["ogg", "mp3", "wav"],
                "optional": True,
            },
        },
        handler=update_voice_settings,
    )
    
    register_tool(
        name="get_voice_duration",
        description="Get the duration of an audio file in seconds.",
        parameters={
            "audio_data": {
                "type": "string",
                "description": "Base64 encoded audio data",
            },
            "format": {
                "type": "string",
                "description": "Audio format",
                "enum": ["ogg", "mp3", "wav", "m4a"],
                "optional": True,
            },
        },
        handler=get_voice_duration,
    )