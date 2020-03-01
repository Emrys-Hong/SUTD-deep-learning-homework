from basics import *

class Flower_dataset(Dataset):
    def __init__(self, image_dir, file_path, transform):
        self.transform = transform
        self.image_dir = Path(image_dir)
        labels = []
        self.image_paths = []
        with open(file_path) as f:
            for line in f.readlines():
                n,l = line.split()
                labels.append(int(l))
                self.image_paths.append(self.image_dir/n)
        self.labels = torch.Tensor(labels).to(torch.int64)
        
    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")
        img.load()
        img = self.transform(img)
        return img, self.labels[i]
    
    def __len__(self):
        return len(self.labels)

class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrain, freeze):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrain)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if freeze:
            for child in self.model.named_children():
                if child == 'fc': continue
                for param in child[1].parameters():
                    param.requires_grad = True

    def forward(self, inputs):
        x = inputs
        return self.model(inputs)

def train(model, dl_trn, optimizer, criterion, num_instances):
    model.train()
    trn_loss, trn_acc = 0, 0
    for X, y in dl_trn:
        inputs = X.to(config.device)
        labels = y.to(config.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        trn_loss += loss.item() * X.size(0)
        trn_acc += torch.sum(preds == labels).item()
    trn_loss /= num_instances
    trn_acc /= num_instances
    return trn_loss, trn_acc

def test(model, dl_val, criterion, num_instances):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for X, y in dl_val:
            inputs = X.to(config.device)
            labels = y.to(config.device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * X.size(0)
            val_acc += torch.sum(preds == labels).item()
    val_loss /= num_instances
    val_acc /= num_instances
    return val_loss, val_acc

def fit(ds_trn, ds_val, ds_tst, dl_trn, dl_val, dl_tst, model, criterion):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=config.MOMENTUM)
    model = model.to(config.device)
    best_acc = 0
    trn_losses, val_losses, val_accs = [], [], []
    for epoch in range(config.NUM_EPOCHS):
        # train
        trn_loss, trn_acc = train(model, dl_trn, optimizer, criterion, len(ds_trn))
        # val
        val_loss, val_acc = test(model, dl_val, criterion, len(ds_val))
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.state_dict()
            best_epoch = epoch
        print(f'Epoch {epoch}/{config.NUM_EPOCHS - 1}, trn_loss {trn_loss} val_loss {val_loss} val_acc {val_acc}')
    return best_weights, best_epoch, [best_acc, trn_losses, val_losses, val_accs]

class config:
    BATCH_SIZE = 64
    NUM_WORKS = 4
    NUM_EPOCHS = 30
    LR = 0.01
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    MOMENTUM = 0.9
    image_dir = "./flowers102/flowers_data/jpg"

def main():
    transform = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    ds_trn = Flower_dataset(config.image_dir, "./flowers102/trainfile.txt", transform)
    ds_val = Flower_dataset(config.image_dir, "./flowers102/valfile.txt", transform)
    ds_tst = Flower_dataset(config.image_dir, "./flowers102/testfile.txt", transform)
    dl_trn = torch.utils.data.DataLoader(ds_trn, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKS)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKS)
    dl_tst = torch.utils.data.DataLoader(ds_tst, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKS)
    
    criterion = nn.CrossEntropyLoss()
    
    print("# first model")
    model = ResNet18(102, pretrain=False, freeze=False)
    best_weights, best_epoch, first_model = fit(ds_trn, ds_val, ds_tst, dl_trn, dl_val, dl_tst, model, criterion)
    model.load_state_dict(best_weights)
    first_model.append(test(model, dl_tst, criterion, len(ds_tst))[1])
    
    print("# second model")
    model = ResNet18(102, pretrain=True, freeze=False)
    best_weights, best_epoch, second_model = fit(ds_trn, ds_val, ds_tst, dl_trn, dl_val, dl_tst, model, criterion)
    model.load_state_dict(best_weights)
    second_model.append(test(model, dl_tst, criterion, len(ds_tst))[1])
    
    print("# third model")
    model = ResNet18(102, pretrain=True, freeze=True)
    best_weights, best_epoch, third_model = fit(ds_trn, ds_val, ds_tst, dl_trn, dl_val, dl_tst, model, criterion)
    model.load_state_dict(best_weights)
    third_model.append(test(model, dl_tst, criterion, len(ds_tst))[1])
    return first_model, second_model, third_model

if __name__ == '__main__':
    first_model, second_model, third_model = main()