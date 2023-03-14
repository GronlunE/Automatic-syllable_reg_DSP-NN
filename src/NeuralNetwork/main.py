# Own implementation
import numpy as np
import pandas as pd
import sys
from config import*
from mat73 import loadmat
from Tensor import build_data
from DSP_NN import run_WaveNet, run_prediciton


def generate_user_commands():
    """

    :return:
    """
    valid_commands = ["basic", "langdep", "datadep", "crossval", "done"]
    commands = []
    epochs = 10
    dims = 32
    batches = 32
    while True:
        command = input("Enter command or type 'done' to start execution: ")

        if command == "done":
            print("Starting execution...")
            break

        if command not in valid_commands:
            print("Invalid command.")
            continue

        if command == "basic":
            commands.append(command)
            print("Added basic operation to command list.")

        elif command == "crossval":
            commands.append(command)
            print("Added crossval operation to command list.")

        elif command == "langdep":
            languages = []
            while True:
                lang = input("Enter language (french, spanish, polish) or type 'done': ")
                if lang == "done":
                    if not languages:
                        print("At least one language must be entered.")
                        continue
                    break
                elif lang not in ["french", "spanish", "polish"]:
                    print("Invalid language.")
                    continue
                languages.append(lang)
            commands.append((command, languages))
            print(
                f"Added language-dependent operation for {len(languages)} "
                f"languages: {', '.join(languages)} to command list.")

        elif command == "datadep":
            n_samples = input("Enter number of samples per language (700, 1425, 2850, 5700): ")
            if n_samples not in ["700", "1425", "2850", "5700"]:
                print("Invalid number of samples.")
                continue
            commands.append((command, n_samples))
            print(f"Added data-dependent operation with {n_samples} samples per language to command list.")

    return commands, epochs, batches, dims


def main():
    """

    :return:
    """
    # orig_stdout = sys.stdout
    # f = open(r"resources\log_test.txt", 'w')
    # sys.stdout = f

    commands, epochs, batches, dims = generate_user_commands()

    # load .mat file
    data = loadmat(tensordata_loc)

    # extract variables from the .mat file
    tensor = data['tensor']
    syllables = np.transpose(data['syllables'])
    labels = data['labels']

    for command in commands:

        if command == "crossval":

            break

        else:

            new_tensor = tensor
            new_syllables = syllables

            if command == "basic":
                # Running everything
                print("Performing the basic test with all data...")
                pass

            elif command[0] == "langdep":
                languages = command[1]
                print(f"Performing language-dependent operation for {len(languages)} languages: {', '.join(languages)}...")

                # Divide tensor and syllables arrays into samples per language
                samples_per_lang = 5700 // len(languages)
                tensor_lang = []
                syllables_lang = []
                for lang in languages:
                    lang_indices = labels[:, 0] == lang
                    tensor_lang.append(tensor[lang_indices][:samples_per_lang])
                    syllables_lang.append(syllables[lang_indices][:samples_per_lang])

                # Concatenate tensors and syllables for all languages
                new_tensor = np.concatenate(tensor_lang, axis=0)
                new_syllables = np.concatenate(syllables_lang, axis=0)

            elif command[0] == "datadep":

                n_samples = int(command[1])

                print(f"Performing data-dependent operation with {n_samples} samples per language...")

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

            # Run Wavenet on new data
            run_WaveNet(new_tensor, new_syllables, epochs=epochs, batch_size=batches, dims=dims)

    # sys.stdout = orig_stdout
    # f.close()


main()
