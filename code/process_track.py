import argparse
import os
import pdb

import mir_eval
import numpy as np
import pandas as pd
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from enhancement import filter_notes_by_key, quantize_notes_to_bpm
from get_key import get_key, keys
from metrics import (calculate_metrics, midi_to_mir_eval_format,
                     mir_eval_format_to_midi)


def process_track(audio_path, midi_target_path, bpm=None, key=None):
    if key is None:
        key = get_key(audio_path)
    if bpm is None:
        pass  # TODO: функцию из bpm_detection добавить
        # bpm = get_bpm(audio_path)

    model_output, predicted_midi, note_events = predict(audio_path)
    target_midi = pretty_midi.PrettyMIDI(midi_target_path)

    # Преобразование MIDI в формат mir_eval
    predicted_notes = midi_to_mir_eval_format(predicted_midi)
    target_notes = midi_to_mir_eval_format(target_midi)

    # Фильтрация нот по ключу
    filtered_notes = filter_notes_by_key(predicted_notes, key)
    filtered_midi = mir_eval_format_to_midi(filtered_notes)

    # Квантизация нот по BPM
    quantized_notes = quantize_notes_to_bpm(predicted_notes, bpm)
    quantized_midi = mir_eval_format_to_midi(quantized_notes)

    # Фильтрация и квантизация нот
    filtered_quantized_notes = quantize_notes_to_bpm(filtered_notes, bpm)
    filtered_quantized_midi = mir_eval_format_to_midi(filtered_quantized_notes)

    # Подсчет F-меры
    _, _, f_measure_original, aor_original = calculate_metrics(predicted_midi, target_midi)
    _, _, f_measure_filtered, aor_filtered = calculate_metrics(filtered_midi, target_midi)
    _, _, f_measure_quantized, aor_quantized = calculate_metrics(quantized_midi, target_midi)
    _, _, f_measure_filtered_quantized, aor_filtered_quantized = calculate_metrics(filtered_quantized_midi, target_midi)

    # Вывод результатов
    print(f"Total Predicted Notes: {len(predicted_notes)}")
    print(f"Total Filtered Notes: {len(filtered_notes)}")
    print(f"Total Quantized Notes: {len(quantized_notes)}")
    print(f"Total Filtered and Quantized Notes: {len(filtered_quantized_notes)}")
    print(f"F-measure (Original Pred vs. Target MIDI): {f_measure_original}")
    print(f"F-measure (Filtered Pred vs. Target MIDI): {f_measure_filtered}")
    print(f"F-measure (Quantized Pred vs. Target MIDI): {f_measure_quantized}")
    print(f"F-measure (Filtered and Quantized Pred vs. Target MIDI): {f_measure_filtered_quantized}")

    print("\n" * 2)

    # Возврат метрик
    return f_measure_original, f_measure_filtered, f_measure_quantized, f_measure_filtered_quantized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tracks")
    parser.add_argument("--df_dataset_path", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.df_dataset_path)

    # Списки для хранения метрик
    original_metrics = []
    filtered_metrics = []
    quantized_metrics = []
    filtered_quantized_metrics = []

    for i, row in df.iterrows():
        if row.get('key') == np.nan:
            continue
        metrics_tuple = process_track(row['audio_path'], row['midi_path'], row.get('bpm'), row.get('key'))
        original_metrics.append(metrics_tuple[0])
        filtered_metrics.append(metrics_tuple[1])
        quantized_metrics.append(metrics_tuple[2])
        filtered_quantized_metrics.append(metrics_tuple[3])

    # Вычисление средних значений метрик
    avg_original = np.mean(original_metrics)
    avg_filtered = np.mean(filtered_metrics)
    avg_quantized = np.mean(quantized_metrics)
    avg_filtered_quantized = np.mean(filtered_quantized_metrics)

    # Вывод средних значений метрик
    print(f"Average F-measure (Original): {avg_original}")
    print(f"Average F-measure (Filtered): {avg_filtered}")
    print(f"Average F-measure (Quantized): {avg_quantized}")
    print(f"Average F-measure (Filtered and Quantized): {avg_filtered_quantized}")