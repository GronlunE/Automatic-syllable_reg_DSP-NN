"""
Data locals
"""

# Epsilon constant
eps = 2.220446049250313e-16

# ----------------------------------------IMPORTANT SYSTEM LOCATION VARIABLES-------------------------------------------

# Testing .mat save locations
resultmat_loc = r"matlab\experiments\results.mat"
datadep_resultmat_loc = r"matlab\experiments\results_datadep.mat"
langdep_resultmat_loc = r"matlab\experiments\results_langdep.mat"

# Where the training data is located.
tensordata_loc = r"resources\tensordata.mat"

# Where the testing data is located.
test_tensordata_loc = r"resources\TEST_tensordata.mat"
english_tensordata_loc = r"resources\english_tensordata.mat"
estonian_tensordata_loc = r"resources\estonian_tensordata.mat"

# CrossFoldVal weights
initial_weights = r"'resource\crossval_initial_model_weights.h5'"
# ---------------------- UNIMPORTANT FOR RUNNING THE NEURAL NETWORK FROM THIS POINT ON ---------------------------------

# Locations for .npz files where the log-Mel arrays are being saved after calculated before further processed.
npz_loc = r"resources\logMel.npz"
estonian_npz_loc = r"resources\estonian_logMel.npz"
english_npz_loc = r"resources\english_logMel.npz"

# CSV save locations.
theta_csv_save_loc = r"resources\csv\thetaSylls.csv"
test_csv_loc = r"C:\Users\Elmeri\PycharmProjects\Automatic-syllable_reg_DSP-NN\src\resources\csv\test_english.csv"

# Original audio file locations.
wav_root = r"resources\audio\**\*.wav"
test_wav_root = r"resources\test_audio\**\*.wav"
english_wav_root = r"resources\test_audio\english\*.wav"
estonian_wav_root = r"resources\test_audio\estonian\*.wav"

# MATLAB location.
matlabroot = r"C:\Program Files\MATLAB\R2022b"

