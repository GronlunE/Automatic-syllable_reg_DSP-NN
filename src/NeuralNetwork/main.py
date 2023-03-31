# Own implementation
import numpy as np
from config import*
from scipy.io import loadmat, savemat
from DSP_NN import run_WaveNet, run_cross_validation, run_prediction, set_baseline


def generate_user_commands():
    """
    Asks the user for the tests
    :return:
    """
    valid_commands = ["basic",
                      "langdep",
                      "datadep",
                      "crossval",
                      "begin",
                      "demo",
                      "baseline",
                      "langdep_full",
                      "datadep_full",
                      "config"]
    commands = []
    epochs = 100
    dims = 32
    batches = 32
    print(f"Epochs: {epochs}\nDims: {dims}\nBatches: {batches}")
    while True:
        command = input(f"Enter command or type 'begin' to start execution: ")

        if command == "begin":
            print("Starting execution...")
            break

        if command == "configure":
            confs_legal = ["epochs", "dims", "batches"]

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
                            elif split[0] == "dims":
                                dims = split[1]
                            elif split[0] == "batches":
                                batches = split[1]

                            print(f"Successfully set {split[0]} to {split[1]}")
                            print(f"Epochs: {epochs}\nDims: {dims}\nBatches: {batches}")
                            continue

                print("Invalid command.")

        if command == "demo":
            commands = [("datadep", "700"),
                        ("langdep", ["french", "spanish", "polish"]),
                        ("crossval", ["estonian", "english"])]

            print(
                f"Added demo operation with commands "
                f": {commands} to command list.")

        if command == "langdep_full":
            commands = ["langdep_full",
                        ("langdep", ["french"]),
                        ("langdep", ["polish"]),
                        ("langdep", ["spanish"]),
                        ("langdep", ["french", "polish"]),
                        ("langdep", ["french", "spanish"]),
                        ("langdep", ["polish", "spanish"]),
                        ("langdep", ["french", "polish", "spanish"])]

            print(
                f"Added demo operation with commands "
                f": {commands} to command list.")

        if command == "datadep_full":
            commands = ["datadep_full",
                        ("datadep", "700"),
                        ("datadep", "1425"),
                        ("datadep", "2850"),
                        ("datadep", "5700")]

            print(
                f"Added demo operation with commands "
                f": {commands} to command list.")

        if command not in valid_commands:
            print("Invalid command.")
            continue

        if command == "basic":
            commands.append(command)
            print("Added basic operation to command list.")

        elif command == "baseline":
            commands.append(command)
            print(f"Added {command} operation to command list.")
            continue

        elif command == "crossval":
            languages = []

            while True:

                lang = input("Enter language (estonian or english) or type 'done': ")

                if lang == "done":
                    if not languages:
                        print("At least one language must be entered.")
                        continue
                    break
                elif lang not in ["estonian", "english"]:
                    print("Invalid language.")
                    continue

                languages.append(lang)

            commands.append((command, languages))

            print(f"Added crossval operation with {', '.join(languages)} to command list.")

        elif command == "langdep":
            languages = []

            while True:

                lang = input("Enter language (french, spanish, polish, all) or type 'done': ")

                if lang == "done":
                    if not languages:
                        print("At least one language must be entered.")
                        continue
                    break
                elif lang == "all":
                    languages = ["french", "spanish", "polish"]
                    break
                elif lang not in ["french", "spanish", "polish"]:
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

    return commands, epochs, batches, dims


def main():
    """

    :return:
    """

    matlab_data = {}
    commands, epochs, batches, dims = generate_user_commands()
    new_tensor = []
    new_syllables = []

    for command in commands:

        cross_val_data = {}
        prediction_data = {}
        history_dict = {}
        model = []
# ----------------------------------------------EXECUTE COMMANDS--------------------------------------------------------

        if command[0] == "crossval":

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

            # load .mat file
            data = loadmat(tensordata_loc)

            # extract variables from the .mat file
            tensor = data['tensor']
            syllables = data['syllables']
            labels = np.transpose(data['labels']).flatten()

            N = tensor.shape[0]

            # Shuffle data
            ord_ = np.arange(N)
            np.random.shuffle(ord_)
            tensor = tensor[ord_, :, :]
            syllables = syllables[ord_]
            labels = labels[ord_]

            if command == "basic":

                # Running everything
                print("Performing the basic test with all data...")
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
                samples_per_lang = 5700 // len(languages)
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

                for lang in ["french", "spanish", "polish"]:

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

            # Run Wavenet on new data
            model, history = run_WaveNet(new_tensor, new_syllables, epochs=epochs, batch_size=batches, dims=dims)

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

            baseline_score = {}
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

    if commands[0] == "langdep_full":
        savemat(langdep_resultmat_loc, matlab_data)

    elif commands[0] == "datadep_full":
        savemat(datadep_resultmat_loc, matlab_data)

    else:
        savemat(resultmat_loc, matlab_data)


main()
