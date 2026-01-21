# Voice Message Error Fix Summary

## Problem
The Telegram voice message integration was failing with the error:
```
messages.0.content.1.document.source.base64.media_type: Input should be 'application/pdf'
```

This occurred because the voice transcription was incorrectly trying to send audio files to Claude API as `document` content type, but Claude's document API only supports PDFs, not audio files.

## Root Cause
1. The `_transcribe_with_claude()` function in `src/gru/tools/voice.py` was sending audio data using `"type": "document"`
2. Claude API's document content type only accepts `application/pdf` media type
3. Audio files (ogg, mp3, wav) were being sent with audio MIME types like `audio/ogg`
4. This caused Claude API to reject the request

## Solution Implemented

### 1. Fixed Claude Transcription Function
- **File**: `src/gru/tools/voice.py`
- **Change**: Replaced the incorrect Claude audio transcription implementation with a proper fallback strategy
- **Details**: 
  - Claude API doesn't actually support direct audio transcription currently
  - Updated function to fall back to OpenAI Whisper API first, then local Whisper as backup
  - Removed the problematic document content type usage

### 2. Updated Default STT Provider
- **File**: `src/gru/tools/voice.py`
- **Change**: Changed default STT provider from `"claude"` to `"openai"`
- **Reason**: Since Claude doesn't support audio transcription, OpenAI Whisper is a better default

### 3. Updated Available Providers Lists
- **Files**: `src/gru/tools/voice.py`, `src/gru/telegram_bot.py`
- **Change**: Removed "claude" from available STT providers in all interfaces
- **Updated**: Tool definitions, help text, validation functions

### 4. Added get_voice_duration Function
- **File**: `src/gru/tools/voice.py`
- **Addition**: Implemented `get_voice_duration()` function as mentioned in the task context
- **Features**: 
  - Supports multiple audio formats (ogg, mp3, wav, m4a)
  - Uses ffprobe if available, falls back to pydub, then estimation
  - Proper error handling and temporary file cleanup

## Files Modified
1. `src/gru/tools/voice.py` - Main voice processing logic
2. `src/gru/telegram_bot.py` - Help text updates

## Testing
Created comprehensive test script that verifies:
- ✅ Default STT provider is no longer Claude
- ✅ Claude removed from available providers
- ✅ Transcription fails gracefully without the document type error
- ✅ All voice tools register correctly

## Next Steps
1. **Restart Gru**: The changes require restarting the Gru server to take effect
   ```bash
   pkill -f gru && cd /Users/zak/gru && .venv/bin/gru-server
   ```

2. **Test with Real Voice Message**: Send a voice message to Telegram bot to verify end-to-end functionality

3. **API Key Configuration**: Ensure OpenAI API key is configured for transcription:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

## Impact
- ✅ Voice messages no longer crash with Claude document error
- ✅ Proper fallback transcription using OpenAI Whisper
- ✅ Better error handling and user feedback
- ✅ Added missing voice duration functionality
- ✅ Maintains backward compatibility for existing voice features

## Error Prevention
- Removed Claude from STT provider options to prevent future misuse
- Added proper validation and error messages
- Implemented graceful fallbacks for transcription services