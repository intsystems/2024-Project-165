import argparse
import os
import random
import subprocess

import pandas as pd
from mido import Message, MetaMessage, MidiFile, MidiTrack
from pretty_midi import (Instrument, KeySignature, Note, PrettyMIDI,
                         TimeSignature)


def get_absolute_path(relative_path):
    return os.path.abspath(relative_path)


def generate_midi(duration, bpm, key_signature, output_dir, index):
    midi_data = PrettyMIDI(initial_tempo=bpm)
    piano_program = 0  # Acoustic Grand Piano
    # ToDo: потом инструмент тоже надо рандомно выбирать
    piano = Instrument(program=piano_program)

    scale_offsets = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
    root_note = 60 + key_signature  # C4 is 60 in MIDI, adjusted by key signature

    # Generating notes within the scale
    scale_notes = [root_note + offset for offset in scale_offsets]

    # Additional octave scales
    scale_notes += [note + 12 for note in scale_notes] + [note - 12 for note in scale_notes if note - 12 >= 0]

    current_time = 0
    total_duration = duration  # Duration in seconds

    # Convert BPM to beat duration in seconds
    beat_duration = 60 / bpm

    while current_time < total_duration:
        pitch = random.choice(scale_notes)
        # pitch = scale_notes[2]  # DEBUG
        start_time = current_time
        # Random note durations: 1, 2, 4 beats (quarter, half, whole notes)
        duration_choices = [beat_duration, beat_duration / 2, beat_duration / 4]  # Quarter, Eighth, Sixteenth notes
        note_duration = random.choice(duration_choices)
        note = Note(
            velocity=100,
            pitch=pitch,
            start=start_time,
            end=start_time + note_duration,
        )
        piano.notes.append(note)
        current_time += note_duration

    midi_data.instruments.append(piano)

    # Adding key signature to the MIDI data
    key = KeySignature(key_number=key_signature, time=0.0)
    midi_data.key_signature_changes.append(key)

    # Save MIDI file
    midi_file = os.path.join(output_dir, f"output_{index}.mid")
    midi_data.write(midi_file)
    return midi_file


def midi_to_audio(midi_file, audio_format, sample_rate=44100):
    soundfont_path = "./generaluser.sf2"
    output_file = midi_file.replace(".mid", f".{audio_format}")
    command = f"fluidsynth -ni {soundfont_path} {midi_file} -F {output_file} -r {sample_rate}"
    subprocess.run(command, shell=True)
    return output_file


def generate_dataset(num_files, duration, audio_format, output_dir):
    midi_dir = os.path.join(output_dir, 'midi')
    audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    data = []

    for i in range(num_files):
        bpm = random.choice(range(60, 180))
        key_signature = random.choice(range(12))  # 12 key shifts within octave
        print(f"Generating file {i + 1}/{num_files}, BPM: {bpm}, Key Signature: {key_signature}")
        midi_file = generate_midi(duration, bpm, key_signature, midi_dir, i + 1)
        audio_file = midi_to_audio(midi_file, audio_format)
        audio_file_path = os.path.join(audio_dir, os.path.basename(audio_file))
        os.rename(audio_file, audio_file_path)
        data.append({
            "audio_path": get_absolute_path(audio_file_path),
            "midi_path": get_absolute_path(midi_file),
            "bpm": bpm,
            "key": key_signature,
        })
        print(f"Generated {midi_file} and {audio_file_path}")

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)
    print(f"Dataset saved to {os.path.join(output_dir, 'dataset.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset of MIDI and audio files.")
    parser.add_argument("--num_files", type=int, default=100, help="Number of MIDI/audio pairs to generate")
    parser.add_argument("--duration", type=int, default=30, help="Duration of each MIDI file in seconds")
    parser.add_argument("--format", choices=['wav', 'mp3'], default='wav', help="Audio format")
    parser.add_argument("--output_dir", type=str, default='output', help="Directory to save generated files")
    args = parser.parse_args()
    generate_dataset(num_files=args.num_files, duration=args.duration, audio_format=args.format,
                     output_dir=args.output_dir)
