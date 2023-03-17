import numpy as np

from Miscellaneous import run_matlab_engine
import pandas as pd
import os.path
import taglib
from config import*


def thetaSegTest(filepaths):
    """

    :param filepaths:
    :return:
    """
    print("Commencing Theta Segmentation...")

    if not os.path.isfile(theta_csv_save_loc):

        eng = run_matlab_engine()
        values = np.array(eng.thetaseg(filepaths))
        filenames = []
        sylls = []

        print("Compiling filenames...")
        for filepath in filepaths:
            filename = filepath.split("\\")[-1].removesuffix(".wav")
            filenames.append(filename)

        print("Compiling syllables...")
        for syllables in values:
            syllables = int(syllables[0])
            sylls.append(syllables)

        print("Compiling Dataframe...")
        thetaSyllDict = {"Filename": filenames, "Syllables": sylls}
        thetaDF = pd.DataFrame(thetaSyllDict)
        thetaDF.to_csv(theta_csv_save_loc, index=False)

    else:

        print("Compiling Dataframe...")
        thetaDF = pd.read_csv(theta_csv_save_loc)

    print("Calculating MA error...")
    testDF = pd.read_csv(test_csv_loc)
    testDictList = testDF.to_dict(orient="records")

    MAE = []
    MAPE = []
    n = 0

    for file in testDictList:
        testSyllables = file["Syllables"]
        testFilename = file["Filename"]

        thetaFileRow = thetaDF.loc[thetaDF["Filename"] == testFilename]
        thetaSyllables = thetaFileRow["Syllables"].values[0]
        MAE.append(np.absolute(testSyllables-thetaSyllables))
        if testSyllables != 0:
            MAPE.append(np.absolute(((testSyllables-thetaSyllables)/testSyllables)*100))
        else:
            print("Zero removed,", "Filename:", testFilename)
        if n % 1000 == 0:
            print(n, "Done")
        n = n + 1

    print("All Done")
    MAE = np.mean(np.array(MAE))
    MAPE = np.mean(np.array(MAPE))

    return MAE, MAPE


def annotate(filepaths):
    """

    :param filepaths:
    :return:
    """
    values = pd.read_csv(r"resources\csv\thetaSyllsTrain.csv")["Syllables"].tolist()
    print("Compiling dict...")
    for filepath in filepaths:
        syllables = int(values[filepaths.index(filepath)])
        print(syllables)
        wav = taglib.File(filepath)
        wav.tags["SYLLABLE_COUNT"] = [str(syllables)]
        wav.save()
