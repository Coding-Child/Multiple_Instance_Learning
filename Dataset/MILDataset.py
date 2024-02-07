import os
import random
from PIL import Image
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from sklearn.model_selection import train_test_split

from util.mask import *
from util.pseudo_label import get_pseudo_label
Image.MAX_IMAGE_PIXELS = None


def collate_fn(batch):
    """
    params:
        batch: list of tuple (image, label, pseudo_label) (batch data)
    """
    images, labels, pseudo_labels = zip(*batch)

    max_patches = max([image.shape[0] for image in images])
    padded_images = [pad_image_to_max_patches(image, max_patches) for image in images]
    padded_pseudo_labels = [pad_pseudo_labels(p_label, max_patches) for p_label in pseudo_labels]

    images_tensor = torch.stack(padded_images)
    pseudo_labels_tensor = torch.stack(padded_pseudo_labels)
    labels_tensor = torch.tensor(labels)

    return images_tensor, labels_tensor, pseudo_labels_tensor


def pad_image_to_max_patches(image, max_patches):
    """
    params:
        image: torch.Tensor (C, H, W) (image tensor)
        max_patches: int (maximum number of patches)
    """
    padding_patches = max_patches - image.shape[0]

    if padding_patches > 0:
        patch_size = image.shape[1:]
        padding = torch.zeros((padding_patches, *patch_size), dtype=image.dtype)

        image = torch.cat([image, padding], dim=0)

    return image


def pad_pseudo_labels(pseudo_labels, max_patches):
    """
    params:
        pseudo_labels: torch.Tensor (pseudo label tensor)
        max_patches: int (maximum number of patches)
    """
    if pseudo_labels.dim() == 0:
        pseudo_labels = pseudo_labels.unsqueeze(0)

    padding_labels = max_patches - len(pseudo_labels)

    if padding_labels > 0:
        padding = torch.zeros(padding_labels, dtype=pseudo_labels.dtype)
        pseudo_labels = torch.cat([pseudo_labels, padding], dim=0)

    return pseudo_labels


def generate_patches_json(path, patch_size, save_path, static, model, n_clusters):
    """
    Generate patches information for a given image and save it to a JSON file.

    :param path: path of the image data
    :param patch_size: The size of the patch (will be the same for height and width)
    :param save_path: path of save the json file
    :param static: static pseudo label
    :model: Deep Neural Network Model for extract the embedding
    :n_clusters: Number of Cluster
    """
    img_label = path.split('/')[1]
    img_name = path.split('/')[-1].split('.')[0]

    img = Image.open(path)
    np_img = np.array(img)

    W, H = img.size
    patches_info = dict()
    patches_list = list()

    # Calculate the number of patches along height and width
    num_patches_height = (H + patch_size - 1) // patch_size  # Ceiling division
    num_patches_width = (W + patch_size - 1) // patch_size  # Ceiling division
    patch_number = 0

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            y = i * patch_size
            x = j * patch_size
            patch_key = f"patch_{patch_number}"
            patch = np_img[y:y + patch_size, x:x + patch_size]

            # Calculate the tissue mask and the percentage of tissue in the patch
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                tissue_mask_patch = tissue_mask(patch)
                tissue_percentage = tissue_percent(tissue_mask_patch)

                # If the tissue percentage is greater than 20%, add to the JSON info
                if tissue_percentage >= 20:
                    patches_info[patch_key] = {
                        "location": (y, x),
                        "size": (patch_size, patch_size)
                    }
                    patches_list.append(Image.fromarray(patch))
            patch_number += 1

    if static:
        if patches_list and int(img_label) == 1:
            pseudo_labels = get_pseudo_label(model, patches_list, n_clusters)
            for i, patch_key in enumerate(patches_info.keys()):
                patches_info[patch_key]['pseudo_label'] = int(pseudo_labels[i])
        elif patches_list and int(img_label) == 0:
            for i, patch_key in enumerate(patches_info.keys()):
                patches_info[patch_key]['pseudo_label'] = 0
    else:
        pseudo_labels = get_pseudo_label(model, patches_list, n_clusters)
        for i, patch_key in enumerate(patches_info.keys()):
            patches_info[patch_key]['pseudo_label'] = int(pseudo_labels[i])

    try:
        json_filename = f'{save_path}/{img_label}/{img_name}.json'
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)  # Ensure the directory exists
        with open(json_filename, 'w') as json_file:
            json.dump(patches_info, json_file, indent=4)
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def extract_patches(json_path, img_path, num_samples):
    """
    params:
        json_path: str (path of json file)
        img_path: str (path of image file)
        num_samples: int (number of samples)
    """
    pseudo_labels = list()

    with open(json_path, 'r') as file:
        patches_info = json.load(file)

    patch_names_with_label_1 = [name for name, data in patches_info.items() if data['pseudo_label'] == 1]
    patch_names_with_label_0 = [name for name, data in patches_info.items() if data['pseudo_label'] == 0]
    sampled_patch_names = []

    if len(patch_names_with_label_1) >= num_samples:
        sampled_patch_names = random.sample(patch_names_with_label_1, num_samples)
    else:
        sampled_patch_names = patch_names_with_label_1
        num_additional_samples = num_samples - len(patch_names_with_label_1)
        
        if num_additional_samples > 0:
            num_additional_samples = min(num_additional_samples, len(patch_names_with_label_0))
            sampled_patch_names += random.sample(patch_names_with_label_0, num_additional_samples)

    with Image.open(img_path) as img:
        patches_list = []

        for patch_name in sampled_patch_names:
            patch_data = patches_info[patch_name]

            x, y = patch_data['location']
            width, height = patch_data['size']
            pseudo_label = patch_data['pseudo_label']

            start_x = x
            start_y = y

            patch = img.crop((start_x, start_y, start_x + width, start_y + height))
    
            patch = np.array(patch)
            mask = tissue_mask(patch)

            patch[~mask] = 0
            patch = Image.fromarray(patch)

            patches_list.append(patch)
            pseudo_labels.append(pseudo_label)

        if not patches_list:
            raise ValueError(f"No patches found path {img_path}")

        return patches_list, pseudo_label


def data_split(csv_file):
    """
    params:
        csv_file: str (path of csv file)
    """
    data_info = pd.read_csv(csv_file)
    train_df, val_test_df = train_test_split(data_info, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df


class PathologyDataset(Dataset):
    def __init__(self, df, num_samples, transform=None):
        """
        params:
            df: pandas.DataFrame (dataframe of csv file)
            num_samples: int (number of samples)
            transform: torchvision. transforms (transform)
        """
        self.data_info = df
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx]['img_path']
        json_path = self.data_info.iloc[idx]['json_path']
        label = self.data_info.iloc[idx]['label']

        patches, pseudo_label = extract_patches(json_path, img_path, self.num_samples)

        if self.transform:
            patches_tensor = torch.stack([self.transform(patch) for patch in patches])
        else:
            patches_tensor = torch.stack([T.ToTensor()(patch) for patch in patches])

        label = torch.tensor(label, dtype=torch.long)
        pseudo_label = torch.tensor(pseudo_label, dtype=torch.long)

        return patches_tensor, label, pseudo_label
