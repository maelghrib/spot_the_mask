import shutil
import zipfile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def unzip_images(data_dir_path, images_dir_path):
    if images_dir_path.exists():
        shutil.rmtree(images_dir_path)

    images_dir_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(data_dir_path / "images.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir_path)

    return images_dir_path


def create_subdirs(images_dir_path):
    train_mask_dir_path = images_dir_path / "train" / "mask"
    train_no_mask_dir_path = images_dir_path / "train" / "no_mask"

    train_mask_dir_path.mkdir(parents=True, exist_ok=True)
    train_no_mask_dir_path.mkdir(parents=True, exist_ok=True)

    test_mask_dir_path = images_dir_path / "test" / "mask"
    test_no_mask_dir_path = images_dir_path / "test" / "no_mask"

    test_mask_dir_path.mkdir(parents=True, exist_ok=True)
    test_no_mask_dir_path.mkdir(parents=True, exist_ok=True)


def move_images_to_subdirs(images_dir_path):
    df = pd.read_csv("./.dataset/train_labels.csv")
    mask_df = df[df["target"] == 1]
    no_mask_df = df[df["target"] == 0]

    train_mask, test_mask = train_test_split(
        mask_df,
        test_size=0.1,
        random_state=42,
    )
    train_no_mask, test_no_mask = train_test_split(
        no_mask_df,
        test_size=0.1,
        random_state=42,
    )

    print(f"Total: {len(mask_df) + len(no_mask_df)}")
    print(f"Total Train: {len(train_mask) + len(train_no_mask)}")
    print(f"Total Test: {len(test_mask) + len(test_no_mask)}")

    for image_path in list(images_dir_path.glob("*")):

        if image_path.name in list(mask_df["image"]):
            if image_path.name in list(train_mask["image"]):
                try:
                    shutil.move(image_path, images_dir_path / "train" / "mask")
                except OSError as e:
                    pass
            elif image_path.name in list(test_mask["image"]):
                try:
                    shutil.move(image_path, images_dir_path / "test" / "mask")
                except OSError as e:
                    pass
        elif image_path.name in list(no_mask_df["image"]):
            if image_path.name in list(train_no_mask["image"]):
                try:
                    shutil.move(image_path, images_dir_path / "train" / "no_mask")
                except OSError as e:
                    pass
            elif image_path.name in list(test_no_mask["image"]):
                try:
                    shutil.move(image_path, images_dir_path / "test" / "no_mask")
                except OSError as e:
                    pass


if __name__ == '__main__':
    data_path = Path(".dataset/")
    images_path = data_path / "images"
    unzip_images(data_path, images_path)
    create_subdirs(images_path)
    move_images_to_subdirs(images_path)
