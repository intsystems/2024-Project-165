import pandas as pd

def quantize_notes_to_bpm(predicted_notes, bpm):
    """
    Quantize note timings based on BPM.

    Parameters:
        predicted_notes (list of tuples): List of note events, where each event is a tuple (start_time, end_time, pitch).
        bpm (float): The tempo of the piece in beats per minute.

    Returns:
        list of tuples: The quantized note events.
    """
    beat_duration = 60 / (bpm * 16) # Duration of a beat in seconds

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


def filter_notes_by_key(predicted_notes, key_index):
    if pd.isnull(key_index):
        print("key index is nan!!!")
        return predicted_notes
    # Scales defined by MIDI note numbers modulo 12 (to match all octaves)
    major_scale_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_scale_intervals = [0, 2, 3, 5, 7, 8, 10]

    # Map each key to its corresponding scale intervals
    key_scales = {i: major_scale_intervals for i in range(12)}  # Major keys
    key_scales.update({i+12: minor_scale_intervals for i in range(12)})  # Minor keys

    # Get the root note of the key (as a MIDI number modulo 12)
    root_note = ((key_index % 12) + 0) % 12

    # Calculate the scale notes for the detected key
    scale_notes = [(note + root_note) % 12 for note in key_scales[key_index]]

    # Filter the predicted notes based on the scale
    filtered_notes = [note for note in predicted_notes if note[2] % 12 in scale_notes]

    return filtered_notes