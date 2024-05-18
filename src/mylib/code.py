import madmom

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

import os

"""
ToDo:
 - создать много файлов с кодом
 - чтобы bpm, key улучшения были функциями от датасета, папки
 - считали и возвращали готовые метрики Fno
 - везде argparse где можно
 - начать делать в пятницу вечером
 - начать с bpm/key 
"""

# Этоп второй, пишем функции метрик и сравниваем
import mir_eval
import pretty_midi


def midi_to_mir_eval_format(midi_data):
    """Преобразование MIDI файла в формат для mir_eval."""
    # midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # mir_eval работает с временем в секундах, pretty_midi возвращает время в секундах
            start = note.start
            end = note.end
            pitch = note.pitch
            notes.append((start, end, pitch))
    return notes


def mir_eval_format_to_midi(note_events, instrument_name='Acoustic Grand Piano'):
    """
    Convert a list of note events to a PrettyMIDI object.

    Parameters:
        note_events (list of tuples): List of note events, where each event is a tuple (start_time, end_time, pitch).
        instrument_name (str): The name of the instrument to use for these notes.

    Returns:
        pretty_midi.PrettyMIDI: The PrettyMIDI object created from the note events.
    """
    # Create a new PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI()

    # Create a new Instrument instance (using a piano program number: 0)
    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)

    # Add notes to the instrument
    for start, end, pitch in note_events:
        # Create a Note instance for each note event
        note = pretty_midi.Note(
            velocity=100,  # Setting a default velocity
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi_data.instruments.append(instrument)

    return midi_data


# # Example usage
# note_events = [
#     (0.5, 1.0, 60),  # C4
#     (1.5, 2.0, 64),  # E4
#     (2.5, 3.0, 67)   # G4
# ]
# midi_file = mir_eval_format_to_midi(note_events)

# # Optionally, save the PrettyMIDI object to a MIDI file
# # midi


def calculate_Fno(predicted_midi, target_midi):
    """Расчет F-measure-no-offset (Fno) между предсказанным и целевым MIDI."""

    # Конвертируем MIDI в формат, подходящий для mir_eval
    predicted_notes = midi_to_mir_eval_format(predicted_midi)
    target_notes = midi_to_mir_eval_format(target_midi)

    if len(predicted_notes) == 0:
        return 0

    # mir_eval требует разделения данных на onset, offset и pitches
    predicted_onsets, predicted_offsets, predicted_pitches = zip(*predicted_notes)
    target_onsets, target_offsets, target_pitches = zip(*target_notes)

    predicted_intervals = np.vstack((predicted_onsets, predicted_offsets)).T
    target_intervals = np.vstack((target_onsets, target_offsets)).T

    # Вычисляем Precision, Recall и F-measure
    # Важно: mir_eval принимает дополнительные параметры для вычисления, например window для сравнения onset
    precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
        target_intervals,  # Интервалы целевых нот
        np.array(target_pitches),  # Частоты целевых нот
        predicted_intervals,  # Интервалы предсказанных нот
        np.array(predicted_pitches),  # Частоты предсказанных нот
        onset_tolerance=0.05,  # Допуск 50 мс для начала ноты
        pitch_tolerance=50,  # Допуск четверти тона (50 центов)
        offset_ratio=None  # Игнорирование окончания ноты
    )

    return f_measure


# Пример вызова
# predicted_midi_path = 'path_to_predicted_midi.mid'
# target_midi_path = 'path_to_target_midi.mid'
# f_no = calculate_Fno(predicted_midi_path, target_midi_path)
# print(f"F-measure-no-offset (Fno): {f_no}")



def filter_notes_by_key(predicted_notes, key_index, keys):
    # Scales defined by MIDI note numbers modulo 12 (to match all octaves)
    major_scale_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_scale_intervals = [0, 2, 3, 5, 7, 8, 10]

    # Map each key to its corresponding scale intervals
    key_scales = {i: major_scale_intervals for i in range(12)}  # Major keys
    key_scales.update({i+12: minor_scale_intervals for i in range(12)})  # Minor keys

    # Get the root note of the key (as a MIDI number modulo 12)
    root_note = ((key_index % 12) + 3) % 12

    # Calculate the scale notes for the detected key
    scale_notes = [(note + root_note) % 12 for note in key_scales[key_index]]

    # Filter the predicted notes based on the scale
    filtered_notes = [note for note in predicted_notes if note[2] % 12 in scale_notes]

    return filtered_notes

key_index = get_key(audio_path)
print(key_index)
filtered_notes = filter_notes_by_key(predicted_notes, key_index, keys)
filtered_midi = mir_eval_format_to_midi(filtered_notes)

print(len(predicted_notes), len(filtered_notes))
print(calculate_Fno(predicted_midi, target_midi))
print(calculate_Fno(filtered_midi, target_midi))


def get_bpm(file_path):
    y, sr = librosa.load(file_path)

    # Estimate the tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    print(f'Estimated BPM: {tempo}')
    return tempo

def quantize_notes_to_bpm(predicted_notes, bpm):
    """
    Quantize note timings based on BPM.

    Parameters:
        predicted_notes (list of tuples): List of note events, where each event is a tuple (start_time, end_time, pitch).
        bpm (float): The tempo of the piece in beats per minute.

    Returns:
        list of tuples: The quantized note events.
    """
    beat_duration = 60 / (bpm * 4) # Duration of a beat in seconds

    quantized_notes = []
    for start, end, pitch in predicted_notes:
        # Quantize the start time to the nearest beat
        quantized_start = round(start / beat_duration) * beat_duration

        # Quantize the end time to the nearest beat
        # Note: This simplistic approach may quantize end times to the same as start times for very short notes.
        quantized_end = round(end / beat_duration) * beat_duration

        # Ensure the quantized end time is at least one quantization step after the start time
        if quantized_end <= quantized_start:
            quantized_end = quantized_start + beat_duration

        quantized_notes.append((quantized_start, quantized_end, pitch))

    return quantized_notes

# Example usage
bpm = get_bpm(audio_path)
print(bpm)
quantized_notes = quantize_notes_to_bpm(predicted_notes, bpm)
quantized_midi = mir_eval_format_to_midi(filtered_notes)

print(len(predicted_notes), len(quantized_notes))
print(calculate_Fno(predicted_midi, target_midi))
print(calculate_Fno(quantized_midi, target_midi))



tracks = os.listdir(dataset_path)
DEBUG_MODE = False
for track in tracks:
    if track.startswith('Track'):
        if 1:
            i = 1
            track_path = os.path.join(dataset_path, track)
            audio_path = os.path.join(track_path, 'stems', f'S0{i}.wav')
            midi_target_path = os.path.join(track_path, 'MIDI', f'S0{i}.mid')

            if os.path.exists(audio_path) and os.path.exists(midi_target_path):
                print(f"Processing {track}...")
                print(audio_path)

                key = get_key(audio_path)
                # bpm = get_bpm(audio_path)

                model_output, predicted_midi, note_events = predict(audio_path)
                target_midi = pretty_midi.PrettyMIDI(midi_target_path)

                tempo_changes = target_midi.get_tempo_changes()
                average_tempo = target_midi.estimate_tempo()
                print("Темп изменений:", tempo_changes[1])  # Показывает все темпы (в BPM), если они меняются
                print(f"Средний темп из midi_target: {average_tempo:.2f}")
                bpm = average_tempo

                # midi_data = converter.parse(midi_target_path)
                # key = midi_data.analyze('key')
                # print(f"Определенный ключ из midi_target: {key.tonic.name} {key.mode}")

                key_signature = None
                for marker in target_midi.key_signature_changes:
                    key_signature = marker
                    break  # Предполагаем, что первое событие смены ключа относится ко всему треку

                if key_signature:
                    print(f"Найденная информация о ключе: {key_signature}")
                else:
                    print("Информация о ключе не найдена.")

                predicted_notes = midi_to_mir_eval_format(predicted_midi)
                target_notes = midi_to_mir_eval_format(target_midi)

                filtered_notes = filter_notes_by_key(predicted_notes, key_index, keys)
                filtered_midi = mir_eval_format_to_midi(filtered_notes)

                quantized_notes = quantize_notes_to_bpm(predicted_notes, bpm)
                quantized_midi = mir_eval_format_to_midi(filtered_notes)

                # Assuming you have functions to calculate F-measure (Fno) and variables holding the appropriate data
                print(f"Total Predicted Notes: {len(predicted_notes)}")
                print(f"Total Filtered Notes: {len(filtered_notes)}")
                print(f"Total Quantized Notes: {len(quantized_notes)}")
                print(f"F-measure (Original Predictions vs. Target MIDI): {calculate_Fno(predicted_midi, target_midi)}")
                print(f"F-measure (Filtered Predictions vs. Target MIDI): {calculate_Fno(filtered_midi, target_midi)}")
                print(
                    f"F-measure (Quantized Predictions vs. Target MIDI): {calculate_Fno(quantized_midi, target_midi)}")

                if DEBUG_MODE:
                    break  # Выход после обработки первого трека, если включен дебаг-режим

                print("\n" * 5)