# SegNet for CamVid Semantic Segmentation

## Description

This project demonstrates a PyTorch implementation of the SegNet model for performing semantic segmentation on the CamVid road scene dataset. 
SegNet is a deep learning architecture that segments images at the pixel level, predicting the object class (e.g., road, car, pedestrian) for each region.

## Dataset Information (CamVid)

CamVid is a dataset of real-world road scene images with pixel-level object class labels.
Each image has a resolution of 480x360 pixels and is labeled with 32 object classes.
This project utilizes the latest CamVid dataset downloaded via kagglehub and splits it into train/val/test sets.
### Data Loading and Preprocessing

The `CamVidDataset` class, inheriting from PyTorch's `Dataset`, is responsible for loading and preprocessing the CamVid dataset.

**Initialization (`__init__`)**:
* It takes the `root_dir` (the main directory of the dataset), `split` (e.g., 'train', 'val'), and an optional `transform` as input.
* It constructs the paths to the image and mask directories based on the `split`.
* It reads the image and mask file names and sorts them to ensure correspondence.
* It loads the `class_dict.csv` file, which maps class names to their RGB color codes.
* It creates dictionaries (`name_to_rgb`, `rgb_to_id`, `id_to_class`, `class_to_id`) to facilitate the conversion between class names, RGB values, and numerical IDs. `num_classes` stores the total number of unique classes.

**Length (`__len__`)**:
* Returns the total number of images in the specified `split`.

**Get Item (`__getitem__`)**:
* Given an `idx`, it retrieves the corresponding image and mask file names.
* It constructs the full paths to the image and mask files.
* It opens the image and mask using PIL (Pillow) and converts them to RGB format.
* If a `transform` is provided (e.g., resizing, normalization), it applies the transformations to both the image and the mask.
* The RGB mask is converted into a **semantic mask** where each pixel value corresponds to the class ID. This is done by iterating through the `rgb_to_id` dictionary and assigning the corresponding class ID to the pixels in the mask that match the RGB color of that class.
* Finally, the semantic mask is converted to a PyTorch `Tensor`.
* It returns the processed image tensor and the semantic mask tensor.

**Transformations**:
The `transform` variable defines a series of image transformations to be applied:
* `transforms.Resize((img_size, img_size))`: Resizes both the image and the mask to a fixed size (256x256 in this case).
* `transforms.ToTensor()`: Converts the PIL Image objects to PyTorch tensors. This also scales the pixel values of the image to the range [0, 1].
* `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: Normalizes the image tensors using the mean and standard deviation typically used for pre-trained models on ImageNet. The mask is not normalized as it contains class IDs.
```
class CamVidDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        image_dir = os.path.join(root_dir, split)
        mask_dir = os.path.join(root_dir, f'{split}_labels')

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('_L.png')])

        class_dict_path = os.path.join(root_dir, 'class_dict.csv')
        self.class_df = pd.read_csv(class_dict_path)
        self.name_to_rgb = {row['name']: (row['r'], row['g'], row['b']) for index, row in self.class_df.iterrows()}
        self.rgb_to_id = {v: index for index, v in enumerate(self.name_to_rgb.values())}
        self.id_to_class = {v: k for k, v in enumerate(self.name_to_rgb.keys())}
        self.class_to_id = {k: v for v, k in self.id_to_class.items()}
        self.num_classes = len(self.class_to_id)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.png', '_L.png')

        img_path = os.path.join(self.root_dir, self.split, img_name)
        mask_path = os.path.join(self.root_dir, f'{self.split}_labels', mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask_np = np.array(mask).astype(np.uint8)
        mask_np = mask_np.transpose(1, 2, 0)

        semantic_mask = np.zeros(mask_np.shape[:2], dtype=np.int64)

        for rgb_tuple, class_id in self.rgb_to_id.items():
            rgb_array = np.array(rgb_tuple, dtype=np.uint8)
            try:
                semantic_mask[(mask_np == rgb_array).all(axis=2)] = class_id
            except ValueError as e:
                raise e

        mask = torch.from_numpy(semantic_mask)

        return image, mask

img_size = 256
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Model Architecture (SegNet)

SegNet is a Fully Convolutional Network with an encoder-decoder structure.
The encoder consists of multiple Conv-BatchNorm-ReLU-MaxPool layers that progressively downsample the input image.
The decoder upsamples using indices obtained from the max-pooling layers (MaxUnpool) to restore the original resolution.
The output of the final decoder layer has a number of channels equal to the number of classes, predicting the class with the highest score for each pixel.
### SegNet Network Architecture

The SegNet model used in this project features a symmetric **Encoder-Decoder** architecture for semantic segmentation.

#### Encoder (SegNetEncoder)
The SegNetEncoder module defines a building block for the encoder part of the network. It consists of two convolutional layers (nn.Conv2d) each followed by Batch Normalization (nn.BatchNorm2d) and ReLU activation (nn.ReLU). The key operation here is the MaxPool2d layer with return_indices=True. This not only downsamples the feature map but also stores the indices of the maximum values within each pooling window. These indices are crucial for the upsampling process in the decoder.

```python
class SegNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x, indices = self.pool(x)
        return x, indices
```
#### Decoder (SegNetDecoder)
The SegNetDecoder module implements the corresponding decoder block. It starts with a MaxUnpool2d layer, which performs the upsampling operation. Unlike simple interpolation methods, MaxUnpool2d uses the indices stored during the max-pooling step in the encoder to place the values in their original maximal locations, resulting in a sparser but more structurally informed upsampling. This is followed by two convolutional layers, each with Batch Normalization and ReLU activation, to refine the upsampled feature maps. The output_size parameter ensures that the output of the unpooling layer has the correct spatial dimensions.
```
class SegNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
```
#### SegNet (Full Model)
The SegNet class assembles the complete network. It consists of a sequence of four SegNetEncoder modules that progressively downsample the input image and extract hierarchical features. The corresponding four SegNetDecoder modules then take these features and the stored pooling indices to upsample them back to the original image resolution. The number of output channels in the final decoder (decoder1) is set to num_classes, which corresponds to the number of object classes in the CamVid dataset. The forward method defines the flow of data through the network: the input x is passed through the encoders, and the resulting feature maps and indices are then passed through the decoders in reverse order, ultimately producing the pixel-wise class predictions. The symmetry between the encoder and decoder paths, along with the use of max-unpooling, is a defining characteristic of the SegNet architecture.
```
class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.encoder1 = SegNetEncoder(in_channels, 64)
        self.encoder2 = SegNetEncoder(64, 128)
        self.encoder3 = SegNetEncoder(128, 256)
        self.encoder4 = SegNetEncoder(256, 512)

        self.decoder4 = SegNetDecoder(512, 256)
        self.decoder3 = SegNetDecoder(256, 128)
        self.decoder2 = SegNetDecoder(128, 64)
        self.decoder1 = SegNetDecoder(64, num_classes)

    def forward(self, x):
        enc1, indices1 = self.encoder1(x)
        enc2, indices2 = self.encoder2(enc1)
        enc3, indices3 = self.encoder3(enc2)
        enc4, indices4 = self.encoder4(enc3)

        dec4 = self.decoder4(enc4, indices4, enc3.size())
        dec3 = self.decoder3(dec4, indices3, enc2.size())
        dec2 = self.decoder2(dec3, indices2, enc1.size())
        dec1 = self.decoder1(dec2, indices1, x.size())

        return dec1
```
## Hyperparameters

* Image Size: 256x256
* Batch Size: 8
* Epochs: 101
* Optimizer: Adam (learning rate=0.001)
* Loss: CrossEntropyLoss
* Device: Automatic selection of CUDA (GPU) or CPU

## Training and Validation Process

The dataset is split into train/val/test sets and loaded using DataLoaders.
In each epoch, the model is trained on the training set and evaluated on the validation set.
The main evaluation metrics are Pixel Accuracy and F1 Score.
Model checkpoints are saved every 5 epochs.
Prediction results are visualized during the training and validation process to assess the model's segmentation performance.

## Improvements and Future Updates

**Current Limitations:**

* Focuses on distinguishing objects from the background within the image rather than detailed per-class performance.
* Limited performance in fine-grained distinctions between objects (classes) and handling complex boundaries.

**Future Directions:**

* Add more diverse evaluation metrics such as per-class IoU (Intersection over Union).
* Apply state-of-the-art techniques such as data augmentation, deeper networks, and attention mechanisms.
* Enhance analysis capabilities with per-class confusion matrices, visualizations, etc.

## Conclusion

This project provides a basic pipeline for semantic segmentation using the SegNet architecture and the CamVid dataset. It can be utilized for pixel-level object recognition in real-world road environments, and further improvements can lead to more precise segmentation performance.
