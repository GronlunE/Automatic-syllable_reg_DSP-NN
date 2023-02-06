# Own implementation
import numpy as np
import taglib

from DSP_NN import run_WaveNet
from Tensor import import_test_mat
import Miscellaneous
import glob
from librosa import get_duration
import pandas as pd

matlabroot = r"C:\Program Files\MATLAB\R2022b"

wav_root = r"resources\audio\**\*.wav"
test_wav_root = r"resources\test_audio\**\*.wav"
test_npz_loc = r"resources\TEST_logMel.npz"
npz_loc = r"resources\logMel.npz"
tensordata_loc = r"resources\tensordata.mat"
test_tensordata_loc = r"resources\TEST_tensordata.mat"


def main():
    language_dirs = glob.glob(r"resources\audio\*", recursive=True)

    npz_data = np.load(npz_loc)
    for filepath in language_dirs:
        csv_dict = {"Filename": [], "Audio duration": [], "LogMel shape": [], "Syllables": []}
        language = filepath.split("\\")[-1]
        syllables = []
        filenames = []
        durations = []
        LM_shapes = []
        directory = filepath + "\\*.wav*"
        for file in glob.glob(directory, recursive=True):
            filename = file.split("\\")[-1]
            syll = int(taglib.File(file).tags["SYLLABLE_COUNT"][0])
            t = round(get_duration(filename=file), 2)
            LM_shape = np.shape(npz_data[filename])[0]

            filenames.append(filename.removesuffix(".wav"))
            LM_shapes.append(LM_shape)
            durations.append(t)
            syllables.append(syll)

            csv_dict["Filename"] = filenames
            csv_dict["Audio duration"] = durations
            csv_dict["Syllables"] = syllables
            csv_dict["LogMel shape"] = LM_shapes

            csv_df = pd.DataFrame(csv_dict)
            csv_df.to_csv(filepath + "\\" + language + ".csv", index=False)
    """
    logMels = Miscellaneous.get_logMel_shapes(npz_loc)
    durations = Miscellaneous.get_audio_durs(wav_root)
    Miscellaneous.compare_dur_and_logMel_shape(logMels,durations)
    """

    import_test_mat(wav_root=test_wav_root,
                    npz_loc=test_npz_loc, 
                    matlabroot=matlabroot,
                    tensordata_loc=test_tensordata_loc)

    """
    # Execute WaveNet
    run_WaveNet(wav_root=wav_root,
                npz_loc=npz_loc,
                tensordata_loc=tensordata_loc,
                test_tensordata_loc= test_tensordata_loc,
                matlabroot=matlabroot,
                epochs=1,
                batch_size=32)
    """


main()
