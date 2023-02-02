# Own implementation
from DSP_NN import run_WaveNet

wav_root = r"resources\audio\**\*.wav"
npz_loc = r"resources\logMel.npz"
tensordata_loc = r"resources\tensordata.mat"
matlab_home = r"C:\Program Files\MATLAB\R2022b"


def main():

    # Execute WaveNet
    run_WaveNet(wav_root=wav_root,
                npz_loc=npz_loc,
                tensordata_loc=tensordata_loc,
                matlab_home=matlab_home,
                epochs=15,
                batch_size=32)

    return


main()
