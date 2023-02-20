# Own implementation
import numpy as np
import pandas as pd
import sys

from DSP_NN import run_WaveNet, run_prediciton
from ThetaSeg import thetaSegTest, annotate
from Miscellaneous import get_filepaths,write_csv
from Tensor import build_data,import_test_mat, build_logMel_npz

matlabroot = r"C:\Program Files\MATLAB\R2022b"

wav_root = r"resources\audio\**\*.wav"
test_wav_root = r"resources\test_audio\**\*.wav"
english_wav_root = r"resources\test_audio\english\*.wav"
estonian_wav_root = r"resources\test_audio\estonian\*.wav"
estonian_npz_loc = r"resources\estonian_logMel.npz"
english_npz_loc = r"resources\english_logMel.npz"
npz_loc = r"resources\logMel.npz"
tensordata_loc = r"resources\tensordata.mat"
test_tensordata_loc = r"resources\TEST_tensordata.mat"
english_tensordata_loc = r"resources\english_tensordata.mat"
estonian_tensordata_loc = r"resources\estonian_tensordata.mat"
theta_csv_save_loc = r"resources\csv\thetaSylls.csv"
test_csv_loc = r"C:\Users\Elmeri\PycharmProjects\Automatic-syllable_reg_DSP-NN\src\resources\csv\test_english.csv"


def main():

    orig_stdout = sys.stdout
    f = open(r"resources\log_test.txt", 'w')
    sys.stdout = f

    """
    MAE, MAPE = thetaSegTest(filepaths=filepaths,
                             matlabroot=matlabroot,
                             theta_csv_save_loc=theta_csv_save_loc,
                             test_csv_loc=test_csv_loc)
    print("Mean Absolute Error:", MAE)
    print("Mean Absolute Percentage Error:", MAPE)
    """

    """
    # Execute WaveNet
    print("\n\nTRAIN: TRAINING DATA | PREDICTION: ESTONIAN, ENGLISH\n\n")
    
    model = run_WaveNet(wav_root=wav_root,
                        npz_loc=npz_loc,
                        tensordata_loc=tensordata_loc,
                        matlabroot=matlabroot,
                        epochs=10,
                        batch_size=16,
                        dims=32)

    run_prediciton(model=model,
                   test_tensordata_loc= estonian_tensordata_loc,
                   batch_size=16)

    run_prediciton(model=model,
                   test_tensordata_loc= english_tensordata_loc,
                   batch_size=16)
    
    print("\n\nTRAIN: TRAINING DATA+ENGLISH | PREDICTION: ESTONIAN\n\n")

    model = run_WaveNet(wav_root=[wav_root, english_wav_root],
                        npz_loc=[npz_loc, english_npz_loc],
                        tensordata_loc= [tensordata_loc, english_tensordata_loc],
                        matlabroot=matlabroot,
                        epochs=10,
                        batch_size=16,
                        dims=32)

    run_prediciton(model=model,
                   test_tensordata_loc= estonian_tensordata_loc,
                   batch_size=16)
    
    print("\n\nTRAIN: ESTONIAN+ENGLISH | PREDICTION: TRAINING\n\n")

    model = run_WaveNet(wav_root=[estonian_wav_root, english_wav_root],
                        npz_loc=[estonian_npz_loc, english_npz_loc],
                        tensordata_loc=[estonian_tensordata_loc, english_tensordata_loc],
                        matlabroot=matlabroot,
                        epochs=10,
                        batch_size=16,
                        dims=32)

    run_prediciton(model=model,
                   test_tensordata_loc=tensordata_loc,
                   batch_size=16)
    """
    print("\n\nTRAIN: ESTONIAN | PREDICTION: ENGLISH\n\n")

    model = run_WaveNet(wav_root=estonian_wav_root,
                        npz_loc=estonian_npz_loc,
                        tensordata_loc=estonian_tensordata_loc,
                        matlabroot=matlabroot,
                        epochs=10,
                        batch_size=16,
                        dims=32)

    run_prediciton(model=model,
                   test_tensordata_loc=english_tensordata_loc,
                   batch_size=16,language="english")

    run_prediciton(model=model,
                   test_tensordata_loc=tensordata_loc,
                   batch_size=16, language="training")

    sys.stdout = orig_stdout
    f.close()

    """
    print("\n\nTRAIN: ENGLISH | PREDICTION: ESTONIAN\n\n")

    model = run_WaveNet(wav_root=english_wav_root,
                        npz_loc=english_npz_loc,
                        tensordata_loc=english_tensordata_loc,
                        matlabroot=matlabroot,
                        epochs=10,
                        batch_size=16,
                        dims=32)

    run_prediciton(model=model,
                   test_tensordata_loc=tensordata_loc,
                   batch_size=16, language="training")

    run_prediciton(model=model,
                   test_tensordata_loc=estonian_tensordata_loc,
                   batch_size=16, language="estonian")
    """
    """
    print("\n\nTRAIN: TRAINING DATA+ESTONIAN | PREDICTION: ENGLISH\n\n")

    model = run_WaveNet(wav_root=[wav_root, estonian_wav_root],
                        npz_loc=[npz_loc, estonian_npz_loc],
                        tensordata_loc= [tensordata_loc, estonian_tensordata_loc],
                        matlabroot=matlabroot,
                        epochs=10,
                        batch_size=8,
                        dims=32)

    run_prediciton(model=model,
                   test_tensordata_loc= english_tensordata_loc,
                   batch_size=8, language="english")
    """
    sys.stdout = orig_stdout
    f.close()


main()
