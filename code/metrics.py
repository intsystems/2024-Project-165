import os

import madmom
# Этап второй, пишем функции метрик и сравниваем
import mir_eval
import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from get_key import get_key, keys


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


def calculate_metrics(predicted_midi, target_midi):
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
    precision, recall, f_measure, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        target_intervals,  # Интервалы целевых нот
        np.array(target_pitches),  # Частоты целевых нот
        predicted_intervals,  # Интервалы предсказанных нот
        np.array(predicted_pitches),  # Частоты предсказанных нот
        onset_tolerance=0.05,  # Допуск 50 мс для начала ноты
        pitch_tolerance=50,  # Допуск четверти тона (50 центов)
        offset_ratio=None  # Игнорирование окончания ноты
    )

    return precision, recall, f_measure, avg_overlap_ratio
