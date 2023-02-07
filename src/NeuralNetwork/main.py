# Own implementation
from DSP_NN import run_WaveNet


matlabroot = r"C:\Program Files\MATLAB\R2022b"

wav_root = r"resources\audio\**\*.wav"
test_wav_root = r"resources\test_audio\**\*.wav"
test_npz_loc = r"resources\TEST_logMel.npz"
npz_loc = r"resources\logMel.npz"
tensordata_loc = r"resources\tensordata.mat"
test_tensordata_loc = r"resources\TEST_tensordata.mat"


def main():

    # Execute WaveNet
    run_WaveNet(wav_root=wav_root,
                npz_loc=npz_loc,
                tensordata_loc=tensordata_loc,
                test_tensordata_loc= test_tensordata_loc,
                matlabroot=matlabroot,
                epochs=10,
                batch_size=32)
    return


main()
