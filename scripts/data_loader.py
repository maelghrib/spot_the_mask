import os
import shutil
import pandas as pd
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split


def remove_old_folders(destination):
    images_folder = destination / "images"
    model_input_folder = destination / "model_input"
    for folder in [images_folder, model_input_folder]:
        if folder.exists():
            shutil.rmtree(folder)


def unzip_file(file, destination):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(destination)


def create_folders(destination):
    images_folder = destination / "images"
    model_input_folder = destination / "model_input"
    train_folder = model_input_folder / "train"
    test_folder = model_input_folder / "test"
    val_folder = model_input_folder / "val"

    for folder in [images_folder, model_input_folder, train_folder, test_folder, val_folder]:
        os.makedirs(folder, exist_ok=True)

    return train_folder, test_folder, val_folder


def split_data(file):
    df = pd.read_csv(file)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}")
    return train_df, test_df, val_df


def move_images(data, source, destination):
    for index, row in data.iterrows():
        image_name = row['image']
        target = "mask" if row['target'] == 1 else "no_mask"

        target_folder = os.path.join(destination, target)
        os.makedirs(target_folder, exist_ok=True)

        try:
            shutil.move(source / image_name, target_folder)
        except OSError as e:
            print(f"Can't find image {source / image_name}")

    print(f"\nTotal of {len(data)} of images moved successfully to {destination}.")
    for dirpath, dirnames, filenames in os.walk(destination):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


if __name__ == '__main__':
    dataset_path = Path("../.dataset/")

    remove_old_folders(destination=dataset_path)

    zip_file = dataset_path / "images.zip"
    unzip_file(file=zip_file, destination=dataset_path)

    train_path, test_path, val_path = create_folders(destination=dataset_path)

    csv_file = dataset_path / "train_labels.csv"
    train, test, val = split_data(file=csv_file)

    images_path = dataset_path / "images"
    move_images(data=train, source=images_path, destination=train_path)
    move_images(data=test, source=images_path, destination=test_path)
    move_images(data=val, source=images_path, destination=val_path)
