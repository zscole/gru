# Telegram Setup

## Create a Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Enter a display name (e.g., "My Gru Bot")
4. Enter a username ending in `bot` (e.g., "my_gru_bot")
5. Copy the token:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

**Optional:** Send `/setprivacy` to @BotFather, select your bot, choose "Disable" to let it read group messages.

## Get Your User ID

1. Search for **@userinfobot** in Telegram
2. Send any message
3. Copy the ID number it returns

Multiple admins: separate IDs with commas (e.g., `123456789,987654321`)

## Environment Variables

```bash
GRU_TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
GRU_ADMIN_IDS=123456789
```

## Voice Messages

Gru supports sending and receiving voice messages through Telegram:

### Receiving Voice Messages
- Send a voice message to your Gru bot
- Gru automatically transcribes it and processes as text
- Supports all major audio formats (OGG, MP3, WAV)

### Sending Voice Messages
```
/gru voice send <chat_id> <text>    # Send TTS to specific chat
/gru voice test "Hello"             # Test TTS in current chat
/gru voice settings                 # View configuration
/gru voice set tts_provider edge    # Change TTS provider
```

### Voice Configuration

**Text-to-Speech Providers:**
- `eleven_labs` - High quality (requires API key)
- `openai` - Good quality (requires API key)
- `edge` - Free Microsoft voices

**Speech-to-Text Providers:**
- `claude` - Uses Claude for transcription (recommended)
- `openai` - OpenAI Whisper API (requires key)
- `whisper` - Local Whisper (slower)

**Environment Variables for Voice:**
```bash
# Optional - for premium TTS
GRU_ELEVEN_LABS_API_KEY=your_key_here
GRU_OPENAI_API_KEY=your_key_here
```

## Troubleshooting

**Bot not responding:**
- Verify `GRU_ADMIN_IDS` matches your user ID exactly
- Check `GRU_TELEGRAM_TOKEN` has no extra spaces
- Restart Gru

**Voice messages not working:**
- Check audio file format is supported (OGG, MP3, WAV)
- Verify API keys if using premium providers
- Try `/gru voice settings` to check configuration

[Back to README](../README.md)
