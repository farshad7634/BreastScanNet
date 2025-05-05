import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BreastCancerSonogramDataset(Dataset):
    """
    Custom Dataset for loading breast sonogram images organized in class subdirectories.
    Expected structure:
        data_root/
            benign/
                img1.png
                img2.png
                ...
            malignant/
                img3.png

                img4.png
                ...
            normal/
                img5.png
                img6.png
                ...
    """
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        # Gather all image paths and labels
        self.samples = []
        classes = sorted(os.listdir(data_root))  # e.g. ['benign', 'malignant', 'normal']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for cls_name in classes:
            cls_folder = os.path.join(data_root, cls_name)
            if not os.path.isdir(cls_folder):
                continue  # Skip non-directory files
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform:
            img = self.transform(img)

      
        return img, label


if __name__ == '__main__':
    # Define transforms: resize, to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale mean/std
    ])

    data_dir = r"E:\Projects\BreastScanNet\BreastScanNet VS code\data"
    dataset = BreastCancerSonogramDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Example: iterate one batch
    imgs, labels = next(iter(dataloader))
    print(f"Batch image tensor shape: {imgs.shape}")  # (16,1,224,224)
    print(f"Batch labels: {labels}")
