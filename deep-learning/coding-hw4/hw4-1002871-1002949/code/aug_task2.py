
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
    
def main():
    transform = transforms.Compose([transforms.FiveCrop(224),
                    lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
                    lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(norm) for norm in norms])])

    dataset = ImageNetDataset('imagenet/imagespart/', 'imagenet/data.csv', crop_size=256, transform=transform)
    dataset_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKS)

    model = models.resnet18(pretrained=True).to(config.device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for batch in dataset_loader:
            batched_fives = batch[0].to(config.device)
            labels = batch[1].to(config.device)

            batch_size, num_crops, c, h, w = batched_fives.size()

            # flatten over batch and five crops
            stacked_fives = batched_fives.view(-1, c, h, w)
            
            result = model(stacked_fives)
            result_avg = result.view(config.BATCH_SIZE, num_crops, -1).mean(1) # avg over crops
            pred = result_avg.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    print(f'Test set: Accuracy: {round(100.*correct/len(dataset_loader.dataset))} %\n')

if __name__ == "__main__":
    main()
