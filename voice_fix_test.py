#!/usr/bin/env python3
"""Test script to verify the voice message fix."""

import asyncio
import base64
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, '/Users/zak/gru/src')

from gru.tools.voice import transcribe_voice_message, get_voice_settings

async def test_voice_fix():
    """Test that voice message processing no longer crashes with Claude document error."""
    print("Testing voice message fix...")
    
    # Test 1: Check that default STT provider is no longer Claude
    settings = await get_voice_settings()
    stt_provider = settings["stt"]["provider"]
    available_providers = settings["stt"]["available_providers"]
    
    print(f"✓ Default STT provider: {stt_provider}")
    print(f"✓ Available STT providers: {available_providers}")
    
    if "claude" in available_providers:
        print("✗ FAIL: Claude still listed in available STT providers")
        return False
    else:
        print("✓ PASS: Claude removed from STT providers")
    
    if stt_provider == "claude":
        print("✗ FAIL: Default provider is still Claude")
        return False
    else:
        print("✓ PASS: Default provider is not Claude")
    
    # Test 2: Try transcription with fake audio data
    # This should NOT crash with the document type error anymore
    fake_audio = base64.b64encode(b"fake_ogg_audio_data").decode()
    
    print("\nTesting transcription fallback...")
    try:
        result = await transcribe_voice_message(fake_audio, "ogg", "claude")
        if "error" in result:
            print(f"✓ PASS: Transcription failed gracefully with: {result['error'][:100]}...")
            # Should not contain the document/PDF error message
            if "application/pdf" in result["error"] or "document.source.base64.media_type" in result["error"]:
                print("✗ FAIL: Still getting Claude document type error")
                return False
            else:
                print("✓ PASS: No Claude document type error")
        else:
            print("? UNEXPECTED: Transcription succeeded with fake data")
            
    except Exception as e:
        error_str = str(e)
        if "application/pdf" in error_str or "document.source.base64.media_type" in error_str:
            print(f"✗ FAIL: Still getting Claude document type error: {error_str}")
            return False
        else:
            print(f"✓ PASS: Failed with different error (expected): {error_str[:100]}...")
    
    print("\n🎉 All tests passed! Voice message fix is working correctly.")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_voice_fix())
    sys.exit(0 if success else 1)