
from basics import *

class config:
    BATCH_SIZE = 4
    NUM_WORKS = 1
    NUM_EPOCHS = 10
    LR = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MOMENTUM = 0.9

class ImageNetDataset(Dataset):
    def __init__(self, image_dir, file_path, transform, crop_size):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.crop_size = crop_size
        csv_file = pd.read_csv(file_path, header=None)
        self.image_names = csv_file.iloc[:,0]
        self.labels = csv_file.iloc[:,1]
        
    def __getitem__(self, i):
        img = Image.open(self.image_dir/self.image_names[i]).convert("RGB")
        img.load()
        h, w = img.size
        scale = self.crop_size / min(w,h)
        new_size = (int(np.ceil(scale*h)), int(np.ceil(scale*w)))
        img = self.transform(img.resize(new_size))
        return img, self.labels[i]

    def __len__(self):
        return len(self.labels)

def main(norm):
    if norm: 
        transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
    
    dataset = ImageNetDataset('imagenet/imagespart/', 'imagenet/data.csv', crop_size=224, transform=transform)
    dataset_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKS)

    model = models.resnet18(pretrained=True).to(config.device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataset_loader):
            images, labels = inputs.to(config.device), labels.to(config.device)
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    print(f'Test set: Accuracy: {round(100. * correct / len(dataset_loader.dataset))}\n')



if __name__ == "__main__":
    print("With Transformation")
    main(True)
    print("Without Transformation")
    main(None)