import os


def data_exploring(PATH_DIR):
    os.chdir(PATH_DIR)

    for folder in os.listdir():
        print(folder)

    labels = []

    for root, dirs, files in os.walk(PATH_DIR):
        print(f"Current directory: {root}")
        label = root.split("/")[-1]
        labels.append(label)
        labels.sort()
    print("--------------------------------")
    if (len(labels)):
        labels.pop(0)
        for label in labels:
            print(label, end=",")

    print(f"Number of labels:{len(labels)} ")

    return labels
