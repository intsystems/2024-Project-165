import os
import csv
import pretty_midi


def get_absolute_path(relative_path):
    return os.path.abspath(relative_path)


def process_track(audio_path, midi_path):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # Extract the tempo (BPM)
    bpm = midi_data.get_tempo_changes()[1][0]

    # Extract the key signature
    if midi_data.key_signature_changes:
        key_signature = midi_data.key_signature_changes[0].key_number
    else:
        key_signature = None

    return bpm, key_signature


def create_dataset(dataset_path, output_csv='dataset.csv'):
    tracks = os.listdir(dataset_path)
    dataset = []

    for track in tracks:
        if track.startswith('Track'):
            for i in range(1, 6):  # Assuming you have 5 stems
                track_path = os.path.join(dataset_path, track)
                audio_path = os.path.join(track_path, 'stems', f'S0{i}.wav')
                midi_target_path = os.path.join(track_path, 'MIDI', f'S0{i}.mid')

                if os.path.exists(audio_path) and os.path.exists(midi_target_path):
                    print(f"Processing {track}...")
                    print(audio_path)

                    bpm, key_signature = process_track(audio_path, midi_target_path)

                    dataset.append({
                        'audio_path': get_absolute_path(audio_path),
                        'midi_path': get_absolute_path(midi_target_path),
                        'bpm': bpm,
                        'key': key_signature
                    })

    # Save dataset to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['audio_path', 'midi_path', 'bpm', 'key'])
        writer.writeheader()
        for data in dataset:
            writer.writerow(data)


if __name__ == "__main__":
    dataset_path = '../datasets/babyslakh_16k'
    output_csv = '../datasets/babyslakh_16k/dataset.csv'
    create_dataset(dataset_path, output_csv)
