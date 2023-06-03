import pandas as pd
import os
import soundfile
import warnings
warnings.filterwarnings("ignore")

import wave

def cut_wav_file(input_file, output_file, start_time, end_time):
    # Open the input .wav file
    data, samplerate = soundfile.read(input_file)
    soundfile.write(input_file, data, samplerate)

    with wave.open(input_file, 'rb') as wav_in:
        # Get the frame rate (samples per second) of the input file
        frame_rate = wav_in.getframerate()
        
        # Calculate the start and end frames based on the given start and end times
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        
        # Set the frame parameters for the output file
        wav_out = wave.open(output_file, 'wb')
        wav_out.setparams(wav_in.getparams())
        
        # Set the position to the start frame
        wav_in.setpos(start_frame)
        
        # Read and write the frames from start to end
        frames_to_write = end_frame - start_frame
        wav_out.writeframes(wav_in.readframes(frames_to_write))
        
        # Close the output file
        wav_out.close()

new_path = 'Data-3/genres_cut_1'
df_train = pd.read_csv('MFML_project/audioset_tagging_cnn/dataset_10_ex/metadata/train.csv', skiprows=3, header=None)
df_eval = pd.read_csv('MFML_project/audioset_tagging_cnn/dataset_10_ex/metadata/eval.csv', skiprows=3, header=None)
df_test = pd.read_csv('MFML_project/audioset_tagging_cnn/dataset_10_ex/metadata/test.csv', skiprows=3, header=None)
# data_train = df_train[0].apply(lambda x: x[:-2])
# data_eval = df_eval[0].apply(lambda x: x[:-2])
# data_test = df_test[0].apply(lambda x: x[:-2])

data_train = df_train[0]
data_eval = df_eval[0]
data_test = df_test[0]



for file in os.listdir('Data-3/genres_original'):
    for wav in os.listdir(f'Data-3/genres_original/{file}'):
        base = os.fsdecode(wav)[:-4]
        
        num_parts = len(data_train[data_train == base])
        mode = 'train'
        if num_parts == 0:
            num_parts = len(data_eval[data_eval == base])
            mode = 'eval'

        if num_parts == 0:
            num_parts = len(data_test[data_test == base])
            mode = 'test'

        if num_parts == 0:
            continue

        else:
            indices = [[30*i, 30*(i+1)] for i in range(num_parts)]
        print(indices)
        counter = 0
        for indice in indices:
            cut_wav_file(f'Data-3/genres_original/{file}/{wav}', f'{new_path}/{mode}/{wav[:-4]}.wav', indice[0], indice[1])
            counter += 1
        

# print(data)