import madmom
import numpy as np
key_processor = madmom.features.key.CNNKeyRecognitionProcessor()


keys = ['A major', 'Bb major', 'B major', 'C major', 'Db major',
          'D major', 'Eb major', 'E major', 'F major', 'F# major',
          'G major', 'Ab major', 'A minor', 'Bb minor', 'B minor',
          'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor',
          'F minor', 'F# minor', 'G minor', 'G# minor']

def get_key(audio_file):
    key_predictions = key_processor(audio_file)
    print(f"Detected key: {np.round(key_predictions, 2)}")
    key_index = np.argmax(key_predictions.flatten())

    detected_key = keys[key_index]
    print(f"Detected key: {detected_key}, Prob: {key_predictions[0, key_index]:.2f}")
    return key_index