"""
Created on Wed Apr 10 16:30:00 2024

@author: GronlunE

PURPOSE:
This script serves as the user inter face for inputting commands to test the functionality of the neural network
defined in models.py.

COMMANDS:
- config:
The command allows to set number of epochs and/or batches for training and/or number of channels for the network.
- langdep:
The command tests the language dependency of the network by regulating the amount of languages in the training data.
- datadep:
The command tests the data dependency of the network by allowing to regulate the amount of data in the training data.
- primary:
The command trains the network with all available data.
- crossval:
Performs cross validation with the testing data (contains actual true targets instead of the DSP derived ones
in the training data). The goal is to evaluate the model's generalization.
- begin:
Executes all listed testing commands.
- end
Immediately exits the program.
"""

import os.path

import numpy as np
from config import*
from scipy.io import loadmat, savemat
from DSP_NN import train_NeuralNet, run_cross_validation, run_prediction, set_baseline
# from Miscellaneous import data_table, csv_to_latex, get_filepaths
# from ThetaSeg import thetaSeg


def interface():
    """
    Function for user inputs.
    :return: List of commands, epochs, batches, channels
    """
    valid_commands = ["primary",
                      "langdep",
                      "datadep",
                      "crossval",
                      "demo",
                      "baseline",
                      "config",
                      "begin",
                      "end"]
    languageNames = ["Dutch", "English", "French", "German", "Italian", "Polish", "Portuguese", "Spanish"]
    commands = []
    epochs = 10
    channels = 32
    batches = 32
    print(f"Epochs: {epochs}\nChannels: {channels}\nBatches: {batches}")
    valid_commands_string = ", ".join(valid_commands)
    print("Valid commands: ")
    print(valid_commands_string)
    while True:
        command = input(f"Enter command or type 'begin' to start execution or 'end' to exit: ")

        if command == "end":
            commands = ["end"]
            return commands, epochs, batches, channels

        if command == "begin":
            if len(commands) == 0:
                # Do something if commands is empty
                print("Commands list is empty, input a testing command before beginning.")
            else:
                break

        if command == "config":
            confs_legal = ["epochs", "channels", "batches"]

            while True:
                conf = input(f"Input one of {confs_legal} and an integer. Separate with space. If done type 'done':")

                if conf == "done":
                    break

                elif conf:
                    split = conf.split(" ")

                    if len(split) == 2:

                        if split[0] in confs_legal and split[1].isdigit():

                            if split[0] == "epochs":
                                epochs = split[1]
                            elif split[0] == "channels":
                                channels = split[1]
                            elif split[0] == "batches":
                                batches = split[1]

                            print(f"Successfully set {split[0]} to {split[1]}")
                            print(f"Epochs: {epochs}\nDims: {channels}\nBatches: {batches}")
                            continue

                print("Invalid command.")

        if command == "demo":
            commands = [("datadep", "700"),
                        ("langdep", ["french", "spanish", "polish"]),
                        ("crossval", ["estonian", "english"])]

            print(
                f"Added demo operation with commands "
                f": {commands} to command list.")

        if command not in valid_commands:
            print("Invalid command.")
            continue

        if command == "primary":
            commands.append(command)
            print("Added the primary operation to command list.")

        elif command == "baseline":
            commands.append(command)
            print(f"Added {command} operation to command list.")
            continue

        elif command == "crossval":
            languages = []

            while True:

                lang = input("Enter language (Estonian or English) or type 'done': ")

                if lang == "done":
                    if not languages:
                        print("At least one language must be entered.")
                        continue
                    break
                elif lang not in ["Estonian", "English"]:
                    print("Invalid language.")
                    continue

                languages.append(lang)

            commands.append((command, languages))

            print(f"Added crossval operation with {', '.join(languages)} to command list.")

        elif command == "langdep":
            languages = []

            while True:

                lang = input(f"Enter language ({', '.join(languageNames)}) or type 'done': ")

                if lang == "done":
                    if not languages:
                        print("At least one language must be entered.")
                        continue
                    break
                elif lang == "all":
                    languages = languageNames
                    break
                elif lang not in languageNames:
                    print("Invalid language.")
                    continue

                languages.append(lang)

            commands.append((command, languages))

            print(
                f"Added language-dependence operation for {len(languages)} "
                f"languages: {', '.join(languages)} to command list.")

        elif command == "datadep":
            n_samples = input("Enter number of samples per language (700, 1425, 2850, 5700): ")

            if n_samples not in ["700", "1425", "2850", "5700"]:
                print("Invalid number of samples.")
                continue

            commands.append((command, n_samples))

            print(f"Added data-dependence operation with {n_samples} samples per language to command list.")

    return commands, epochs, batches, channels


def main():
    """

    :return:
    """

    matlab_data = {}
    commands, epochs, batches, dims = interface()
    new_tensor = []
    new_syllables = []

    for command in commands:

        cross_val_data = {}
        prediction_data = {}
        baseline_score = {}
        history_dict = {}
        model = []
# ----------------------------------------------EXECUTE COMMANDS--------------------------------------------------------

        if commands[0] == "end":
            print("Exiting...")
            return

        elif command[0] == "crossval":

            languages = command[1]

            for language in languages:
                print(f"Performing cross validation for {language}...")
                true_mae, true_mape, dsp_mae, dsp_mape = run_cross_validation(language=language)
                cross_val_data[language] = {"DSP_MAE": dsp_mae,
                                            "DSP_MAPE": dsp_mape,
                                            "True_MAE": true_mae,
                                            "True_MAPE": true_mape}

        elif command == "langdep_full" or command == "datadep_full":
            continue

        elif command == "baseline":
            languages = ["estonian", "english"]
            baseline_score = {}

            for language in languages:
                mae, mape = set_baseline(language)
                baseline_score[language] = {"MAE": mae, "MAPE": mape}

        else:

            # load .mat files

            languageNames = ["Dutch", "English", "French", "German", "Italian", "Polish", "Portuguese", "Spanish"]
            tensor = np.empty((0, 300, 12))
            syllables = np.empty((0, 1))
            labels = np.empty((0, 1))

            for language in languageNames:

                path = os.path.join(new_traindata_loc, language + ".mat")
                data = loadmat(path)

                # extract variables from the .mat file
                tensor_tmp = data['tensor']
                syllables_tmp = np.transpose(data['syllables'])
                labels_tmp = np.full((10000, 1), language)

                N = tensor_tmp.shape[0]

                # Shuffle data
                ord_ = np.arange(N)
                np.random.shuffle(ord_)
                tensor_tmp = tensor_tmp[ord_, :, :]
                syllables_tmp = syllables_tmp[ord_]
                labels_tmp = labels_tmp[ord_]

                tensor = np.concatenate((tensor, tensor_tmp), axis=0)
                syllables = np.concatenate((syllables, syllables_tmp), axis=0)
                labels = np.concatenate((labels, labels_tmp), axis=0)

            # Find the indices of the NaN values in the tensor
            nan_indices = np.argwhere(np.isnan(tensor))

            # Remove the NaN values from the tensor and update the label and syllable arrays
            tensor = np.delete(tensor, nan_indices[:, 0], axis=0)
            syllables = np.delete(syllables, nan_indices[:, 0], axis=0)
            labels = np.delete(labels, nan_indices[:, 0], axis=0)

            # Shuffle data
            N = tensor.shape[0]
            ord_ = np.arange(N)
            np.random.shuffle(ord_)
            tensor = tensor[ord_, :, :]
            syllables = syllables[ord_]
            labels = labels[ord_]
            labels = labels[:,0]

            """
            tensor = tensor[0:1000, :, :]
            syllables = syllables[0:1000]
            labels = labels[0:1000]
            """
            del tensor_tmp, syllables_tmp, labels_tmp

            if command == "primary":

                # Running everything
                print("Performing the primary test with all data...")
                new_tensor = tensor
                new_syllables = syllables
                pass

            elif command == "baseline":
                languages = ["estonian", "english"]
                baseline_score = {}

                for language in languages:
                    mae, mape = set_baseline(language)
                    baseline_score[language] = {"MAE": mae, "MAPE": mape}

            elif command[0] == "langdep":
                languages = command[1]
                print(f"Performing language-dependence test for {len(languages)} "
                      f"languages: {', '.join(languages)}...")

                # Divide tensor and syllables arrays into samples per language
                num_langs = len(languages)
                if num_langs == 1:
                    samples_per_lang = 5700
                else:
                    samples_per_lang = 5700 // 2  # default for two languages
                    if num_langs == 3:
                        samples_per_lang = 5700 // 3

                tensor_lang = []
                syllables_lang = []
                for lang in languages:
                    lang_indices = np.array(labels[:]) == lang
                    tensor_lang.append(tensor[lang_indices][:samples_per_lang])
                    syllables_lang.append(syllables[lang_indices][:samples_per_lang])

                # Concatenate tensors and syllables for all languages
                new_tensor = np.concatenate(tensor_lang, axis=0)
                new_syllables = np.concatenate(syllables_lang, axis=0)

            elif command[0] == "datadep":

                n_samples = int(command[1])

                print(f"Performing data-dependence test with {n_samples} samples per language...")

                # Create new arrays for data dependent on n_samples
                new_tensor = []
                new_syllables = []

                labels = np.char.strip(labels)

                for lang in languageNames:
                    print(tensor.shape)
                    print(labels.shape)
                    print(lang)
                    lang_data = tensor[labels == lang]
                    lang_syllables = syllables[labels == lang]

                    # Take n_samples from lang_data
                    lang_data = lang_data[:n_samples]
                    lang_syllables = lang_syllables[:n_samples]

                    # Append to new arrays
                    new_tensor.append(lang_data)
                    new_syllables.append(lang_syllables)

                # Concatenate new arrays
                new_tensor = np.concatenate(new_tensor)
                new_syllables = np.concatenate(new_syllables)

            # Free space
            del data, tensor, syllables, labels

            # Run model on new data
            model, history = train_NeuralNet(new_tensor, new_syllables, epochs=epochs, batch_size=batches, dims=dims)

            history_dict = {'Loss': history.history['loss'],
                            'MAE': history.history['mean_absolute_error'],
                            "MAPE": history.history['mean_absolute_percentage_error'],
                            "Val_Loss": history.history['val_loss'],
                            "Val_MAE": history.history['val_mean_absolute_error'],
                            "Val_MAPE": history.history['val_mean_absolute_percentage_error'],
                            "Epochs": epochs}

# ---------------------------------------------ORGANIZING THE DATA------------------------------------------------------

        if command[0] == "crossval":

            # Save crossval results
            matlab_data[f"Command_{commands.index(command) + 1}"] = {"Call": np.array(command, dtype=object),
                                                                     "Languages": cross_val_data}

        elif command == "baseline":

            matlab_data[f"Command_{commands.index(command) + 1}"] = {"Call": command,
                                                                     "Languages": baseline_score}

        else:
            languages = ["estonian", "english"]

            # Run predictions
            for language in languages:
                mae, mape = run_prediction(model=model, batch_size=32, language=language)
                prediction_data[language] = {"MAE": np.array([mae]), "MAPE": np.array([mape])}

            # Save results
            matlab_data[f"Command_{commands.index(command) + 1}"] = {"Call": np.array(command, dtype=object),
                                                                     "History": history_dict,
                                                                     "Predictions": prediction_data}

            # Delete unnecessary variables
            del model

# ---------------------------------------------SAVING THE DATA----------------------------------------------------------

    if commands[0] == "primary":
        savemat(primary_resultmat_loc, matlab_data)

    elif commands[0] == "langdep_full":
        savemat(langdep_resultmat_loc, matlab_data)

    elif commands[0] == "datadep_full":
        savemat(datadep_resultmat_loc, matlab_data)

    elif commands[0][0] == "crossval":
        savemat(crossval_resultmat_loc, matlab_data)

    else:
        savemat(resultmat_loc, matlab_data)


main()
