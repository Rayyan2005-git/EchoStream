"""Audio Device Diagnostic and Peripheral Detection Tool for EchoStream.

Lists all available audio input and output devices and tests standard 16kHz
sampling capability.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pyaudio


def list_audio_devices() -> None:
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")

    print("\n================ AVAILABLE AUDIO DEVICES ================")
    print(f"Total Audio Devices Detected: {numdevices}\n")

    input_devices = []
    output_devices = []

    for i in range(0, numdevices):
        dev_info = p.get_device_info_by_host_api_device_index(0, i)
        max_in = dev_info.get("maxInputChannels", 0)
        max_out = dev_info.get("maxOutputChannels", 0)
        name = dev_info.get("name")
        rate = int(dev_info.get("defaultSampleRate", 0))

        if max_in > 0:
            input_devices.append((i, name, max_in, rate))
        if max_out > 0:
            output_devices.append((i, name, max_out, rate))

    print("--- [INPUT DEVICES (Microphones)] ---")
    for idx, name, ch, rate in input_devices:
        print(f"  [Index {idx}] {name} (Channels: {ch}, Default Rate: {rate}Hz)")

    print("\n--- [OUTPUT DEVICES (Speakers/Headphones)] ---")
    for idx, name, ch, rate in output_devices:
        print(f"  [Index {idx}] {name} (Channels: {ch}, Default Rate: {rate}Hz)")

    # Test Default Input & Output at 16000Hz
    print("\n--- [PERIPHERAL COMPATIBILITY CHECK] ---")
    try:
        default_in = p.get_default_input_device_info()
        print(f"Default Input Device : {default_in['name']} (Index {default_in['index']})")
    except Exception as e:
        print(f"Default Input Device Warning: {e}")

    try:
        default_out = p.get_default_output_device_info()
        print(f"Default Output Device: {default_out['name']} (Index {default_out['index']})")
    except Exception as e:
        print(f"Default Output Device Warning: {e}")

    # Check 16kHz format support
    try:
        supported = p.is_format_supported(
            rate=16000,
            input_device=default_in["index"],
            input_channels=1,
            input_format=pyaudio.paInt16,
        )
        print("16kHz 16-bit Mono Input Check: ✅ Supported")
    except Exception as e:
        print(f"16kHz 16-bit Mono Input Check: ⚠️ Notice ({e})")

    p.terminate()
    print("=========================================================\n")


if __name__ == "__main__":
    list_audio_devices()

