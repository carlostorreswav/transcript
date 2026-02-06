#!/usr/bin/env python3
"""
🎙️ Transcriptor de Audio - Carlos Edition
Transcribes audio files using OpenAI Whisper.
Sends macOS notifications when done.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import whisper

# Directories
SCRIPT_DIR = Path(__file__).parent.resolve()
AUDIO_DIR = SCRIPT_DIR / "audio"
OUTPUT_DIR = SCRIPT_DIR / "transcriptions"
HISTORICAL_DIR = SCRIPT_DIR / "historical"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".opus"}

# Ratio aproximado: tiempo real de proceso / duración del audio
# Basado en tests: 5:13 de audio → 2:07 de proceso ≈ 0.4x
PROCESS_RATIO = 0.4


def notify(title: str, message: str, sound: bool = True):
    """Send macOS notification"""
    sound_cmd = 'sound name "Glass"' if sound else ""
    script = f'display notification "{message}" with title "{title}" {sound_cmd}'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def format_time(seconds: int) -> str:
    """Format seconds as MM:SS"""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using ffprobe"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except:
        # Fallback: estimate from file size (rough: 1MB ≈ 6 seconds for WAV)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb * 6


def show_progress_bar(stop_event: threading.Event, file_name: str, estimated_seconds: float):
    """Animated progress bar with ETA"""
    bar_width = 30
    idx = 0
    
    while not stop_event.is_set():
        elapsed = idx
        
        if estimated_seconds > 0:
            progress = min(elapsed / estimated_seconds, 0.99)  # Never show 100% until done
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            percent = int(progress * 100)
            eta = max(0, estimated_seconds - elapsed)
            
            sys.stdout.write(f"\r   [{bar}] {percent:2d}% • {format_time(elapsed)} / ~{format_time(estimated_seconds)}")
        else:
            # Fallback: just show spinner and time
            chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            sys.stdout.write(f"\r   {chars[idx % len(chars)]} Transcribiendo... {format_time(elapsed)}")
        
        sys.stdout.flush()
        time.sleep(1)
        idx += 1
    
    # Final state
    bar = "█" * bar_width
    sys.stdout.write(f"\r   [{bar}] ✅ Completado en {format_time(idx)}      \n")
    sys.stdout.flush()
    return idx


def main():
    # Suppress FP16 warning
    import warnings
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
    
    # Create directories
    for d in [AUDIO_DIR, OUTPUT_DIR, HISTORICAL_DIR]:
        d.mkdir(exist_ok=True)

    # Find audio files
    audio_files = sorted([
        f for f in AUDIO_DIR.iterdir() 
        if f.suffix.lower() in AUDIO_EXTENSIONS
    ])

    if not audio_files:
        print("━" * 50)
        print("⚠️  No hay archivos de audio en la carpeta 'audio/'")
        print(f"   Formatos: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        print("━" * 50)
        notify("Transcriptor", "No hay archivos de audio para procesar")
        return

    print("━" * 50)
    print("🎙️  TRANSCRIPTOR DE AUDIO")
    print("━" * 50)
    print(f"\n📁 {len(audio_files)} archivo(s) encontrados:\n")
    
    for f in audio_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        duration = get_audio_duration(f)
        print(f"   • {f.name} ({size_mb:.1f} MB, ~{format_time(duration)})")

    # Load Whisper model
    print("\n🧠 Cargando modelo Whisper (medium)...")
    model = whisper.load_model("medium")
    print("✅ Modelo listo\n")

    notify("Transcriptor", f"Iniciando transcripción de {len(audio_files)} archivo(s)")

    total_time = 0
    results = []

    for i, audio_file in enumerate(audio_files, 1):
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📝 [{i}/{len(audio_files)}] {audio_file.name}")

        # Get audio duration and estimate processing time
        audio_duration = get_audio_duration(audio_file)
        estimated_process_time = audio_duration * PROCESS_RATIO

        # Start progress thread
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=show_progress_bar, 
            args=(stop_event, audio_file.stem, estimated_process_time)
        )
        progress_thread.daemon = True

        start_time = time.time()
        progress_thread.start()

        # Transcribe
        result = model.transcribe(str(audio_file), language="es")

        stop_event.set()
        progress_thread.join()

        duration = int(time.time() - start_time)
        total_time += duration

        # Save transcription
        output_file = OUTPUT_DIR / f"{audio_file.stem}.txt"
        text = result["text"].strip()
        output_file.write_text(text)
        
        word_count = len(text.split())
        print(f"   💾 {output_file.name} ({word_count} palabras)")

        # Move to historical
        historical_file = HISTORICAL_DIR / audio_file.name
        # Handle duplicates
        if historical_file.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            historical_file = HISTORICAL_DIR / f"{audio_file.stem}_{timestamp}{audio_file.suffix}"
        
        shutil.move(str(audio_file), str(historical_file))
        print(f"   📦 → historical/{historical_file.name}")

        results.append({
            "name": audio_file.stem,
            "words": word_count,
            "time": duration
        })

    # Summary
    print("\n" + "━" * 50)
    print("🎉 RESUMEN")
    print("━" * 50)
    
    total_words = sum(r["words"] for r in results)
    print(f"\n   📄 {len(results)} archivo(s) transcritos")
    print(f"   📝 {total_words:,} palabras totales")
    print(f"   ⏱️  Tiempo total: {format_time(total_time)}")
    print(f"\n   📁 Transcripciones: transcriptions/")
    print(f"   📦 Audios movidos:  historical/")
    print("\n" + "━" * 50)

    # Final notification
    notify(
        "✅ Transcripción Completa",
        f"{len(results)} archivo(s) • {total_words:,} palabras • {format_time(total_time)}"
    )

    # Open transcriptions folder
    subprocess.run(["open", str(OUTPUT_DIR)])


if __name__ == "__main__":
    main()
