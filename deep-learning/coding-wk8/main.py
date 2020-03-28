from basic_imports import *
from torch_imports import *
import glob
import unicodedata
import string

class Config:
    datadir = 'data/names/*.txt'
    batch_size = 16
    hidden_dim = 128
    lr = 0.03
    device = 'cuda:1'
    num_layers = 1
    num_epochs = 7
    

def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, all_letters) for line in lines]

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter, all_letters):
    return all_letters.find(letter)

def lineToTensor(lines, all_letters):
    word_list = []
    for word in lines:
        letter_list = []
        for l in word:
            letter_list.append(letterToIndex(l, all_letters))
        word_list.append(letter_list)
    return np.array(word_list)


def collate_fn(batch):
    x, y = zip(*batch)
    lens = np.array([len(o) for o in x])
    max_len = max(lens)
    seq_lens_idx = np.argsort(lens)[::-1]
    lens = torch.Tensor(lens[seq_lens_idx]).long()
    x = [x[i] for i in seq_lens_idx]
    tensor_y = torch.Tensor([y[i] for i in seq_lens_idx]).long()
    tensor_x = torch.zeros( len(x), max_len, n_letters) 
    for wi, word in enumerate(x):
        for li, letter in enumerate(word):
            tensor_x[wi][li][letter] = 1
            
    return tensor_x, lens, tensor_y

class NameDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self): return len(self.y)

    


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model_type):
        super().__init__()
        if model_type == 'rnn':
            self.model = nn.RNN
        elif model_type == 'lstm':
            self.model = nn.LSTM
        elif model_type == 'gru':
            self.model = nn.GRU
        self.rnn = self.model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.model_type = model_type
        
    def forward(self, inputs):
        x, seq_lens = inputs
        ## TODO: need to unpack?
        x_pack = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)
        if self.model_type == 'rnn' or self.model_type == 'gru':
            y_pack, hn = self.rnn(x_pack)
        elif self.model_type == 'lstm':
            y_pack, (hn, _) = self.rnn(x_pack)
        y, _ = nn.utils.rnn.pad_packed_sequence(y_pack, batch_first=True)
        y_last = torch.stack([y[i,seq_lens[i]-1,:] for i in range(len(x))])
        output = self.fc(y_last)
        return output
    


def train(model, dl, optimizer, criterion, num_instances, config, train=True):
    if train:
        model.train()
    else:
        model.eval()
    loss, acc = 0, 0
    
    for X, seq_lens, y in dl:
        X = X.to(config.device)
        seq_lens = seq_lens.to(config.device)
        labels = y.to(config.device)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(train):
            outputs = model( (X, seq_lens) )
            preds = outputs.argmax(dim=1)
            l = criterion(outputs, labels)
            if train:
                l.backward()
                optimizer.step()
        loss += l.item()
        acc += torch.sum(preds == labels).item()
    loss /= len(dl)
    acc /= num_instances
    return loss, acc

def fit(ds_trn, ds_val, dl_trn, dl_val, model, criterion, config):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    best_acc = 0
    trn_losses, val_losses, val_accs = [], [], []
    for epoch in range(config.num_epochs):
        # train
        trn_loss, trn_acc = train(model, dl_trn, optimizer, criterion, len(ds_trn), config, train=True)
        # val
        val_loss, val_acc = train(model, dl_val, optimizer, criterion, len(ds_val), config, train=False)
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.state_dict()
            best_epoch = epoch
        print(f'Epoch {epoch}/{config.num_epochs - 1}, trn_loss {trn_loss} val_loss {val_loss} val_acc {val_acc}')
    return best_weights, best_epoch, [best_acc, trn_losses, val_losses, val_accs]


def main():
    global n_letters
    config = Config()
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    for filename in findFiles(config.datadir):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename, all_letters)
        category_lines[category] = lines
    n_categories = len(category_lines.keys())

    max_len = max([len(o) for k in category_lines.keys() for o in category_lines[k]])
    trn_xs, trn_ys, val_xs, val_ys = [], [], [], []

    for k in category_lines.keys():
        names = category_lines[k]
        x = lineToTensor(names, all_letters)
        idx = np.arange(len(names))
        y = np.array([all_categories.index(k) for _ in range(len(names))])
        ## train and tst
        trn_idx, tst_idx = np.split(idx, [int(len(names)*0.8)])
        trn_x, trn_y, val_x, val_y = x[trn_idx], y[trn_idx], x[tst_idx], y[tst_idx]

        ## merge
        trn_xs += list(trn_x)
        trn_ys.append(trn_y)
        val_xs += list(val_x)
        val_ys.append(val_y)
    trn_ys = np.concatenate(trn_ys)
    val_ys = np.concatenate(val_ys)
    
    ds_trn = NameDataset(trn_xs, trn_ys)
    ds_val = NameDataset(val_xs, val_ys)
    dl_trn = DataLoader(ds_trn, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    model = RNN(n_letters, config.hidden_dim, config.num_layers, n_categories, 'lstm')
    model = model.to(config.device)
    best_weights, best_epoch, [best_acc, trn_losses, val_losses, val_accs] = fit(ds_trn, ds_val, dl_trn, dl_val, model, criterion, config)
    return best_weights, best_epoch, [best_acc, trn_losses, val_losses, val_accs]

if __name__ == '__main__':
    best_weights, best_epoch, [best_acc, trn_losses, val_losses, val_accs] = main()
    print(f"Best accuracy of current configuration is: {best_acc}")