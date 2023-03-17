# Own implementation
import numpy as np
from config import*
from scipy.io import loadmat, savemat
from DSP_NN import run_WaveNet, run_cross_validation, run_prediciton


def generate_user_commands():
    """

    :return:
    """
    valid_commands = ["basic", "langdep", "datadep", "crossval", "begin", "demo"]
    commands = []
    epochs = 1
    dims = 32
    batches = 32
    while True:
        command = input("Enter command or type 'begin' to start execution: ")

        if command == "begin":
            print("Starting execution...")
            break

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

        if command == "basic":
            commands.append(command)
            print("Added basic operation to command list.")

        elif command == "baseline":
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
    for command in commands:

        cross_val_data = {}
        prediction_data = {}

        if command[0] == "crossval":

            languages = command[1]

            for language in languages:
                print(f"Performing cross validation for {language}...")
                dsp_fs, true_fs = run_cross_validation(language=language)
                cross_val_data[language] = {"DSP_Fscore": dsp_fs, "True_Fscore": true_fs}

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
                continue

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

            else:

                new_tensor = tensor
                new_syllables = syllables

            # Free space
            del data, tensor, syllables, labels

            # Run Wavenet on new data
            model, history = run_WaveNet(new_tensor, new_syllables, epochs=epochs, batch_size=batches, dims=dims)
            weights = model.get_weights()

            history_dict = {'Loss': history.history['loss'],
                            'MAE': history.history['mean_absolute_error'],
                            "MAPE": history.history['mean_absolute_percentage_error'],
                            "Val_Loss": history.history['val_loss'],
                            "Val_MAE": history.history['val_mean_absolute_error'],
                            "Val_MAPE": history.history['val_mean_absolute_percentage_error']}

            if command[0] == "crossval":

                # Save crossval results
                matlab_data[f"Command {commands.index(command)}"] = {"Call": command, "Languages": cross_val_data}

            elif command == "baseline":

                baseline_score = {}
                matlab_data[f"Command {commands.index(command)}"] = {"Call": command, "Languages": baseline_score}

            else:
                languages = ["estonian", "english"]

                # Run predictions
                for language in languages:
                    mae, mape = run_prediciton(model=model, batch_size=32, language=language)
                    prediction_data[language] = {"MAE": np.array([mae]), "MAPE": np.array([mape])}

                # Save results
                matlab_data[f"Command_{commands.index(command) + 1}"] = {"Call": np.array(command),
                                                                         "Weights": np.array(weights, dtype=object),
                                                                         "History": history_dict,
                                                                         "Predictions": prediction_data}

            # Delete unnecessary variables

        del model

    savemat(resultmat_loc, matlab_data)


main()
