import os
import subprocess
import random
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, DataLoader
import config # Contains ash2txt username and password

# Read config file
with open('data.json') as f:
    CONFIG = json.load(f)

DATA = CONFIG["data"]

# Append a directory prefix to the data locations. This allows for custom locations on different computers.
prefix = CONFIG["data_prefix"]

# Set the paths of the fragment data to include the prefix
for item in CONFIG["data"].values():
    item['parent_path'] = prefix + item['parent_path']
    if 'label_path' in item:
        item['label_path'] = prefix + item['label_path']

# Load actions from config file
ACTIONS = CONFIG["actions"]

# Initialize wandb with your project name
wandb.init(project="full_scroll_ink")

# Loop through a directory of 16bit TIFs and rotate them by a given angle, resizing the image so no part of the image 
def rotate_and_resize_tif_files(directory_path, angle):
    for filename in os.listdir(directory_path):
        if filename.endswith('.tif'):
            full_path = os.path.join(directory_path, filename)

            # Construct the command
            cmd = [
                'convert',              # Use convert to create a new image
                '-limit', 'memory', '10GiB',  # Set memory limit 
                '-limit', 'disk', '15GiB',   # Set disk limit
                full_path,              # Input file
                '-depth', '16',         # Specify bit depth
                '-colorspace', 'Gray',  # Set colorspace to grayscale
                '-type', 'Grayscale',   # Specify the image type as grayscale
                '-background', 'black', # Set background color to black
                '-rotate', str(angle),  # Rotate by the given angle
                '-extent', '100%',      # Ensure the image is not shrunk
                full_path               # Output file (overwrites original)
            ]

            # Execute the command
            result = subprocess.run(cmd, stderr=subprocess.PIPE)

            # Check for errors
            if result.returncode != 0:
                print(f"Error rotating {filename}: {result.stderr.decode()}")
            else:
                print(f"Rotated {filename} by {angle} degrees")

# Download a set of files from the ash2txt server
def download_files(user, password, base_url, custom_id, save_dir, postfix, post_download_rotation=None):
    # Incorporate the custom ID into the URL
    url = base_url + str(custom_id) + (postfix if postfix is not None else "")

    # wget command
    cmd = [
        "wget",
        "--no-parent",
        "-r",
        "-l",
        "1",
        "--user=" + user,
        "--password=" + password,
        url,
        "-np",
        "-nd",
        "-nc",  # Don't overwrite files
        "-P",
        save_dir
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        error_output = e.output.decode('utf-8')  # Decode the error output as UTF-8
        print(f"Command failed with exit code {exit_code}.\nError output:\n{error_output}")
    finally:
        # Define the rotation lockfile path
        lockfile_path = os.path.join(os.path.dirname(save_dir), "rotation.lock")

        # Check if the lockfile already exists, so we don't rotate the images twice
        if post_download_rotation and not os.path.exists(lockfile_path):
            rotate_and_resize_tif_files(save_dir, post_download_rotation)

            # Create the lockfile after rotating
            with open(lockfile_path, 'w') as lockfile:
                lockfile.write('Rotation performed')

# Crop the raw fragment data
def crop_images_folder(src, dest, dimensions, finetune_rotation):
    # https://chat.openai.com/c/3319fb1e-54f6-4072-a6f8-1d46f37ac19a
    def sort_key(path):
        return int(os.path.splitext(os.path.basename(path))[0])
    
    image_files = glob.glob(src + "/*.tif")
    image_files = sorted(image_files, key=sort_key)

    # Check if correct number of images already exist in the dest folder. If so skip redoing the cropping
    existing_files = glob.glob(dest + "/*.tif")
    if len(existing_files) == len(image_files[dimensions[2]:dimensions[5]]):
        # Check if one of the images has the correct dimensions
        sample_image = Image.open(existing_files[0])
        if sample_image.size == (dimensions[4] - dimensions[1], dimensions[3] - dimensions[0]):
            return

    all_images = []
    
    for image_file in image_files[dimensions[2]:dimensions[5]]:
        pil_image = Image.open(image_file)
        pil_image = (np.array(pil_image) / 255).astype(np.uint8)

        # Apply final fine tuning rotation so the papyrus is horizontal/vertical
        larger_image = Image.fromarray(pil_image[dimensions[0]:dimensions[3], dimensions[1]:dimensions[4]])

        # Rotate the larger image
        larger_image = larger_image.rotate(finetune_rotation, center=(larger_image.size[0] / 2, larger_image.size[1] / 2))

        all_images.append(np.array(larger_image))

    all_images = np.array(all_images)

    # Save the images to the dest folder
    os.makedirs(dest, exist_ok=True)

    for idx, img in enumerate(all_images):
        pil_image = Image.fromarray(img)
        pil_image.save(os.path.join(dest, f'{idx}.tif'), format='TIFF')

# Copy images from one folder to another and convert to .tif. Used to create labels directory
def copy_image_files(src, dest):
    # Get all .tif files in the source directory
    image_files = glob.glob(src + "/*.tif")
    
    # Make sure the destination directory exists
    os.makedirs(dest, exist_ok=True)
    
    # Iterate through all image files
    for image_file in image_files:
        # Get the file name from the path
        file_name = os.path.basename(image_file)
        
        # Create the destination path with the .png extension
        dest_file_path = os.path.join(dest, os.path.splitext(file_name)[0] + ".png")

        # Check if the file already exists in the destination
        if not os.path.exists(dest_file_path):
            # Open the source image
            img = Image.open(image_file)

            # Convert the image to greyscale
            img_greyscale = img.convert("L")

            # Save the greyscale image as a .png file in the destination directory
            img_greyscale.save(dest_file_path)
            print(f"Copied and converted {file_name} to greyscale .png in {dest}")
        else:
            print(f"Skipped {file_name} as it already exists in {dest}")

# Download & prepare all fragment data
for fragment_data in DATA.values():
    fragment_download_path = os.path.join(fragment_data["parent_path"], "raw_fragments", str(fragment_data["id"]))
    fragment_download_layers_path = os.path.join(fragment_data["parent_path"], "raw_fragments", str(fragment_data["id"]), fragment_data["layers_path"] if "layers_path" in fragment_data.keys() else "layers")
    fragment_cropped_save_path = os.path.join(fragment_data["parent_path"], "cropped_fragments", str(fragment_data["id"]), "_".join(map(str, fragment_data["crop"])), fragment_data["layers_path"] if "layers_path" in fragment_data.keys() else "layers")
    fragment_labels_save_path = os.path.join(fragment_data["parent_path"], "labels", str(fragment_data["id"]), "_".join(map(str, fragment_data["crop"])))
    os.makedirs(fragment_download_layers_path, exist_ok=True)
    
    # Download fragment metadata
    download_files(
        user = config.username,
        password = config.password,
        base_url = fragment_data["url"],
        custom_id = fragment_data["id"],
        save_dir = fragment_download_path,
        postfix = None,
    )

    # Download fragment layers to layers/
    download_files(
        user = config.username,
        password = config.password,
        base_url = fragment_data["url"],
        custom_id = fragment_data["id"],
        save_dir = fragment_download_layers_path,
        postfix = f'/{fragment_data["layers_path"] if "layers_path" in fragment_data.keys() else "layers"}',
        post_download_rotation = fragment_data["post_download_rotation"] if "post_download_rotation" in fragment_data.keys() else None
    )

    crop_images_folder(
        src = fragment_download_layers_path,
        dest = fragment_cropped_save_path,
        dimensions = fragment_data["crop"],
        finetune_rotation = 0 # Unused
    )

    copy_image_files(
        src = fragment_cropped_save_path,
        dest = fragment_labels_save_path
    )

# A dataset of a single volume from a single fragment
class VolumeDataset(Dataset):
    def __init__(self, root_dir, label_path=None, transform=None, tio_transform=None, step=(1,1,1), subvolume_depth=1, subvolume_height=100, subvolume_width=100, is_16bit=False):
        self.root_dir = root_dir
        self.transform = transform
        self.tio_transform = tio_transform
        self.label_path = label_path
        self.step = step
        self.subvolume_depth = subvolume_depth
        self.subvolume_height = subvolume_height
        self.subvolume_width = subvolume_width
        self.image_files = sorted(glob.glob(self.root_dir + "/*.tif"), key=self.sort_key)
        self.label_files = sorted(glob.glob(self.label_path + "/*.png"), key=self.sort_key) if self.label_path else None
        self.is_16bit = is_16bit
        self.load_volume()

    # Used to sort filenames
    def sort_key(self, path):
        return int(os.path.splitext(os.path.basename(path))[0])

    def load_volume(self):
        # Get the shape of the first image to determine the dimensions of the volume
        sample_image = Image.open(self.image_files[0])
        height, width = sample_image.size

        # Create an empty volume with the appropriate shape
        depth = len(self.image_files)
        self.volume = np.empty((depth, width, height), dtype=np.uint8)

        for idx, image_file in tqdm(enumerate(self.image_files), desc="Loading dataset"):
            # Load image
            image = Image.open(image_file)

            # Scale 16bit images down to an 8bit range
            if self.is_16bit:
                image = (np.array(image) / 255).astype(np.uint8)

            # Add image to the volume
            self.volume[idx] = np.array(image)

        # Load labels
        if self.label_files:
            label_stack = []
            for label_file in self.label_files:
                label = Image.open(label_file)
                label_stack.append(np.array(np.array(label) > 0, dtype=np.uint8))
            self.labels = np.stack(label_stack, axis=0)
        else:
            self.labels = None # there are no labels

    def __len__(self):
        depth, height, width = self.volume.shape
        step_d, step_h, step_w = self.step
        return ((depth-self.subvolume_depth)//step_d) * ((height-self.subvolume_height)//step_h) * ((width-self.subvolume_width)//step_w)

    def __getitem__(self, idx):
        depth, height, width = self.volume.shape
        step_d, step_h, step_w = self.step
        
        # TODO I don't think this will work correctly with depth >1, but that's fine for now
        steps_d = (depth-self.subvolume_depth) // step_d
        steps_h = (height-self.subvolume_height) // step_h
        steps_w = (width-self.subvolume_width) // step_w
        d = (idx // (steps_h * steps_w)) * step_d
        h = ((idx % (steps_h * steps_w)) // steps_w) * step_h
        w = (idx % steps_w) * step_w

        sub_volume = self.volume[d:d+self.subvolume_depth, h:h+self.subvolume_height, w:w+self.subvolume_width]
        sub_volume = sub_volume.squeeze(0) # Remove depth dimension

        # Apply transforms
        if self.transform:
            sub_volume = self.transform(sub_volume)
            sub_volume = sub_volume.squeeze(0) # Remove depth dimension

        if self.tio_transform:
            sub_volume = self.tio_transform(sub_volume.unsqueeze(0)).squeeze()

        # Add the channel dimension
        sub_volume = sub_volume[None, :, :]

        # Right now, the model label is a float from 0 to 1, which scales linearly such that a 0 label means that there is labeled ink anywhere in the sampled sub-volume, and a 1 means that 50% or more of the sampled sub-volume is labeled as ink
        if self.labels is not None:
            labels_image = self.labels[d:d+self.subvolume_depth, h:h+self.subvolume_height, w:w+self.subvolume_width]
            label = np.clip(labels_image.sum() * 2 / (self.subvolume_depth * self.subvolume_height * self.subvolume_width), 0.0, 1.0)
        else:
            label = 0.0 # If there are no labels

        position = (d, h, w)  # position of the sampled sub-volume

        return sub_volume, label, position

# A dataset of many volumes from many fragments
class MultiVolumeDataset:
    def __init__(self, directories, label_paths=None, transform=None, tio_transform=None, step=(1,1,1), subvolume_depth=1, subvolume_height=100, subvolume_width=100, is_16bit=False):
        self.volumes = [
            VolumeDataset(root_dir=dir, label_path=label_path, transform=transform, tio_transform=tio_transform, step=step, subvolume_depth=subvolume_depth, subvolume_height=subvolume_height, subvolume_width=subvolume_width, is_16bit=is_16bit)
            for dir, label_path in zip(directories, label_paths or [None] * len(directories))
        ]

    def __len__(self):
        return sum(len(volume) for volume in self.volumes)

    def __getitem__(self, idx):
        for volume in self.volumes:
            if idx < len(volume):
                return volume[idx]
            idx -= len(volume)
        raise IndexError('Index out of range')

# 90 degree image rotation for transforms
def random_90_rotation(image):
    rotations = [0, 90, 180, 270]
    angle = random.choice(rotations)
    return image.rotate(angle)

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),                       # Convert NumPy array to PIL Image
    transforms.Lambda(lambda x: x.convert("RGB")), # Convert grayscale to RGB
    transforms.RandomHorizontalFlip(),             # 50% chance of flipping horizontally
    transforms.RandomVerticalFlip(),               # 50% chance of flipping verticallytransforms.RandomRotation(degrees=30),              # Random rotation within a range of ±30 degrees
    transforms.RandomApply([
        transforms.RandomRotation(degrees=15),     # Random rotation within a range of ±15 degrees
    ], p=0.60),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3)     # Add brightness jitter with a factor of 0.3
    ], p=0.5),                                     
    transforms.RandomApply([
        transforms.ElasticTransform(alpha=5.0, sigma=3.0, 
                                interpolation=InterpolationMode.BILINEAR, fill=0), # Elastic transformation
    ], p=0.33),
    transforms.Lambda(random_90_rotation),         # Apply random 90-degree rotation
    transforms.Grayscale(num_output_channels=1),   # Convert back to grayscale
    transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.456], std=[0.227]), # Guesstimated from imagenet RGB presets
])

# Used for inference
transform_just_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.456], std=[0.227]),
])

# Unused (for now)
tio_transform = tio.Compose([
    tio.RandomFlip(axes=(0, 2)),
    tio.OneOf({
        tio.RandomAffine(scales=(1.0, 1.1),degrees=0): 0.33,
        tio.RandomNoise(mean=0, std=0.02): 0.33,
        tio.RandomBlur(std=[0.1, 1.0]): 0.33
    }, p=0.33),
    tio.RandomElasticDeformation(
        num_control_points=5,
        locked_borders=2,
        p=0.33,
    )
])

# Set deterministic seeds
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# Model definition
def create_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 1)
    return model
model = create_model()

# Check if a GPU is available and if not, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # Force using CPU (sometimes good for debugging)
model = model.to(device)

# Wrap model and optimizer with wandb
wandb.watch(model)

# Define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 

# Actions are defined in data.json
for action in ACTIONS:
    if action["action"] == "train":
        fragment_datas = action["data"]

        dirs = []
        label_paths = []

        # Get a list of directories from which to load fragment data and labels
        for fragment_name in action["data"]:
            fragment_data = DATA[fragment_name]
            dirs.append(
                os.path.join(fragment_data["parent_path"], "cropped_fragments", str(fragment_data["id"]), "_".join(map(str, fragment_data["crop"])), fragment_data["layers_path"] if "layers_path" in fragment_data.keys() else "layers")
            )
            label_paths.append(
                os.path.join(fragment_data["parent_path"], "labels", str(fragment_data["id"]), "_".join(map(str, fragment_data["crop"]))),
            )

        # Create dataset
        dataset = MultiVolumeDataset(
            directories=dirs,
            label_paths=label_paths,
            transform=transform
        )

        # Get the indices for all samples in dataset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        # Randomly shuffle the indices
        np.random.shuffle(indices)

        # Calculate the number of samples for divisor of data
        split = int(np.floor(1 / (action["dataset_divisor"]) * dataset_size))

        # Get the indices for the divisor split
        subset_indices = indices[:split]

        # Create a DataLoader with the SubsetRandomSampler
        dataloader = DataLoader(dataset, batch_size=64, sampler=SubsetRandomSampler(subset_indices))

        # Get 20^2 random indices for training samples image
        num_images = 20
        indices = np.random.choice(len(dataset), size=num_images**2, replace=False)

        # Initialize a figure 
        fig = plt.figure(figsize=(20,20))

        for i in range(num_images**2):
            # Get the image and label
            image, label, _ = dataset[indices[i]]
            
            # Add subplot
            ax = fig.add_subplot(num_images, num_images, i+1)
            ax.imshow(image.squeeze(), cmap='gray')
            ax.set_title(f'Label: {label}')
            ax.axis('off')

        # Save the figure 
        plt.tight_layout()
        plt.savefig('training_grid.png')

        # Training loop
        for epoch in range(action["epochs"]):  # loop over the dataset multiple times
            running_loss = 0.0

            pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for i, data in pbar:
                # Get the inputs and labels
                inputs, labels, _ = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Convert labels to float type
                labels = labels.float().to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels.view(-1))  # Labels and outputs need to be 1D for BCEWithLogitsLoss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() # TODO do a real filter or something

                # Log the loss to wandb
                wandb.log({
                    "loss": running_loss/(i+1),
                    "epoch": epoch,
                    "batch": i,
                })

                # Update progress bar
                pbar.set_description(f'Epoch: {epoch+1}, Running Loss: {running_loss/(i+1):.8f}')
            
            pbar.close()

            # Print average loss per batch after each epoch
            print(f'\nEnd of Epoch: {epoch+1}, Average Loss: {running_loss/len(dataloader):.3f}\n')

        print('Finished Training')

    elif action["action"] == "infer":
        fragment_data = DATA[action["data"]]

        model.eval()

        torch.cuda.empty_cache() # Was having out of memory issues and this seemed to help

        dataset = None

        layers_dir = ""

        # If the infer on all key is provided, then ignore the cropped version of the fragment and load the whole, uncropped fragment 
        if "infer_on_all" in action.keys():
            layers_dir = os.path.join(DATA[action["data"]]["parent_path"], "raw_fragments", str(DATA[action["data"]]["id"]), fragment_data["layers_path"] if "layers_path" in fragment_data.keys() else "layers")

        else:
            layers_dir = os.path.join(DATA[action["data"]]["parent_path"], "cropped_fragments", str(DATA[action["data"]]["id"]), "_".join(map(str, DATA[action["data"]]["crop"])), fragment_data["layers_path"] if "layers_path" in fragment_data.keys() else "layers")
        
        dataset = VolumeDataset(layers_dir,
                            DATA[action["data"]]["label_path"] if "label_path" in DATA[action["data"]] else None,
                            transform=transform_just_to_tensor,
                            step=action["step"],
                            is_16bit="infer_on_all" in action.keys())


        if "infer_on_all" in action.keys():
            # Set the start and end dims to be the size of the whole volume
            d_start = 0
            h_start = 0
            w_start = 0
            d_end = dataset.volume.shape[0]
            h_end = dataset.volume.shape[1]
            w_end = dataset.volume.shape[2]
        else:
            # Get original data dimensions
            h_start, w_start, d_start, h_end, w_end, d_end = DATA[action["data"]]["crop"]

        # Subvolume (model input) sizes
        depth = d_end - d_start - 1
        height = h_end - h_start - 100
        width = w_end - w_start - 100

        # Calculate output sizes based on step size# Calculate output sizes based on step size
        d_out = (depth // action["step"][0]) + 1
        h_out = (height // action["step"][1]) + 1
        w_out = (width // action["step"][2]) + 1

        # Create properly sized output array on the GPU with dtype uint8
        outputs = torch.zeros((d_out, h_out, w_out), dtype=torch.uint8, device=device).to(device)

        dataloader = DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True)

        # Iterate through dataloader
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for i, (inputs, labels, pos) in pbar:

            # Skip the totally empty areas on the edges
            if inputs.max() == 0.0:
                continue

            # Move to device
            inputs = inputs.to(device)
            
            # Get predictions
            with torch.no_grad():
                preds = model(inputs)

            # Update progress bar
            pbar.set_description(f'Running inference on {action["data"]}')

            # Compute indices in outputs array
            d_idx = (pos[0] // action["step"][0]).long()
            h_idx = (pos[1] // action["step"][1]).long()
            w_idx = (pos[2] // action["step"][2]).long()

            # Apply sigmoid, scale, and convert to uint8
            scaled_preds = (torch.sigmoid(preds) * 255).to(torch.uint8)

            # Insert predictions into output array using faster indexing
            outputs[d_idx, h_idx, w_idx] = scaled_preds.squeeze(1)

        pbar.close()

        # Move outputs to CPU and convert to a NumPy array
        outputs = outputs.cpu().numpy()

        # Calculate the mean and maximum across the layers for each pixel
        mean_image = np.mean(outputs, axis=0).astype('uint8')
        max_image = np.max(outputs, axis=0).astype('uint8')

        # Create the directory structure
        base_dir = f'tmp/{fragment_data["id"]}_{action["run_name"]}' if "run_name" in action.keys() else f'tmp/{fragment_data["id"]}/'
        raw_layers_dir = os.path.join(base_dir, 'raw_layers')
        os.makedirs(raw_layers_dir, exist_ok=True)

        # Save the mean and maximum images
        mean_image_pil = Image.fromarray(mean_image)
        max_image_pil = Image.fromarray(max_image)
        mean_image_pil.save(os.path.join(base_dir, 'mean_image.png'))
        max_image_pil.save(os.path.join(base_dir, 'max_image.png'))

        wandb.log({
            f"{fragment_data['id']}_mean_image": wandb.Image(mean_image),
            f"{fragment_data['id']}_max_image": wandb.Image(max_image)
        })

        # Save each layer of the outputs
        for i in range(outputs.shape[0]):
            image_pil = Image.fromarray((outputs[i,:,:]).astype('uint8'))
            image_pil.save(os.path.join(raw_layers_dir, f'output_image_{i}.png'))
            wandb.log({
                f"{fragment_data['id']}_output_image_{i}": wandb.Image(outputs[i, :, :].astype('uint8'))
            })
            
        # Calculate and save max for adjacent layers with different group sizes
        for group_size, folder_name in [(2, '2_max'), (3, '3_max'), (5, '5_max'), (10, '10_max')]:
            os.makedirs(os.path.join(base_dir, folder_name), exist_ok=True)
            for i in range(0, outputs.shape[0] - group_size + 1, group_size):
                max_image_adj = np.max(outputs[i:i+group_size,:,:], axis=0).astype('uint8')
                image_pil = Image.fromarray(max_image_adj)
                image_pil.save(os.path.join(base_dir, folder_name, f'max_image_{i}.png'))
                wandb.log({
                    f"{group_size}_step_max_image": wandb.Image(outputs[i, :, :].astype('uint8'))
                })

        print("Saved results to", base_dir)

        print('Done running on', fragment_data)

        torch.cuda.empty_cache() 

    elif action["action"] == "save":
        torch.save(model.state_dict(), action["filename"])

    elif action["action"] == "load":
        model.load_state_dict(torch.load(action["filename"]))
