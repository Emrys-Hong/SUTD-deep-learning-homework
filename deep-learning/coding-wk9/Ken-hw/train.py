import copy
from pathlib import Path
from typing import List, Dict, Tuple

import fire
import numpy as np
import torch
import torch.utils.data
from sklearn import model_selection
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from tcn import TCNWrapper


class HyperParams:
    def __init__(
        self,
        root="data",
        lr=1e-3,
        bs=32,
        steps_per_epoch=1000,
        epochs=100,
        num_hidden=128,
        model_name="lstm",
        num_layers=1,
        seq_len=32,
        dev_run=False,
    ):
        self.root = root
        self.lr = lr
        self.bs = bs
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.num_hidden = num_hidden
        self.model_name = model_name
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dev_run = dev_run
        print(self.__class__.__name__, self.__dict__)


class Splits:
    train = "train"
    val = "val"
    test = "test"

    @classmethod
    def check_valid(cls, x: str) -> bool:
        return x in {cls.train, cls.val, cls.test}


class Vocab:
    pad = "<pad>"
    start = "<start>"
    end = "<end>"
    unk = "<unk>"

    def __init__(self, items: List[str], use_special_tokens=True):
        self.special = []
        if use_special_tokens:
            self.special = [self.pad, self.start, self.end, self.unk]

        unique = self.special + sorted(set(items))
        self.stoi = {s: i for i, s in enumerate(unique)}
        self.itos = {i: s for i, s in enumerate(unique)}
        print(dict(vocab=len(self)))

    def __len__(self) -> int:
        assert len(self.stoi) == len(self.itos)
        return len(self.stoi)

    def encode(self, items: List[str]) -> List[int]:
        return [self.stoi.get(s, self.stoi[self.unk]) for s in items]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.itos[i] for i in indices]


def shuffle_multi_split(
    items: list, fractions=(0.8, 0.1, 0.1), seed=42, eps=1e-6
) -> list:
    assert abs(sum(fractions) - 1) < eps
    assert len(fractions) > 0
    if len(fractions) == 1:
        return [items]

    part_first, part_rest = model_selection.train_test_split(
        items, train_size=fractions[0], random_state=seed
    )
    parts_all = [part_first] + shuffle_multi_split(
        part_rest, normalize(fractions[1:]), seed
    )
    assert len(items) == sum(map(len, parts_all))
    return parts_all


class StarTrekCharGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, hparams: HyperParams, data_split: str, sep_line="\n"):
        assert Splits.check_valid(data_split)
        self.hparams = hparams
        self.root = Path(self.hparams.root)
        self.data_split = data_split
        self.sep_line = sep_line

        self.lines = self.download()
        self.vocab = Vocab(list(self.sep_line.join(self.lines)))
        self.text = self.train_val_test_split()
        self.tensor = self.get_sequences()
        self.show_samples()

    def download(self) -> List[str]:
        url = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/star_trek_transcripts_all_episodes.csv"
        path = self.root / Path(url).name
        if not path.exists():
            download_url(url, str(self.root), filename=path.name)

        with open(path) as f:
            f.readline()  # Skip header
            return [line.strip().strip(",") for line in f]

    def train_val_test_split(self, fractions=(0.8, 0.1, 0.1)) -> str:
        indices_all = list(range(len(self.lines)))
        indices_split = shuffle_multi_split(indices_all, fractions)
        indices = indices_split[
            [Splits.train, Splits.val, Splits.test].index(self.data_split)
        ]
        lines = [self.lines[i] for i in indices]
        text = self.sep_line.join(lines)
        print(dict(lines=len(lines), text=len(text)))
        return text

    def get_sequences(self) -> torch.Tensor:
        path_cache = self.root / f"cache_tensor_{self.data_split}.pt"
        token_start = self.vocab.stoi[self.vocab.start]

        if not path_cache.exists():
            encoded = self.vocab.encode(list(self.text))
            sequences = []
            for i in tqdm(range(len(encoded) - self.hparams.seq_len)):
                sequences.append([token_start] + encoded[i : i + self.hparams.seq_len])
            tensor = torch.from_numpy(np.array(sequences)).type(torch.long)
            torch.save(tensor, str(path_cache))

        tensor = torch.load(str(path_cache))
        print(dict(tensor=tensor.shape))
        return tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, i):
        sequence = self.tensor[i, :]
        return sequence[:-1], sequence[1:]

    def sequence_to_text(self, sequence: torch.Tensor):
        assert sequence.ndim == 1
        return "".join(self.vocab.decode(sequence.numpy()))

    def show_samples(self, num=3):
        print(dict(show_samples=num))
        indices = np.random.choice(len(self), size=num, replace=False)
        for i in indices:
            sequence = self.tensor[i, :]
            print(dict(text=self.sequence_to_text(sequence), raw=sequence))


def normalize(items: list) -> list:
    total = sum(items)
    return [item / total for item in items]


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dict(device=device))
    return device


class SequenceNet(torch.nn.Module):
    def __init__(
        self, num_vocab: int, num_labels: int, hparams: HyperParams, batch_first=True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.embed = torch.nn.Embedding(num_vocab, hparams.num_hidden)
        selector = dict(lstm=torch.nn.LSTM, gru=torch.nn.GRU, tcn=TCNWrapper)
        self.net = selector[hparams.model_name](
            input_size=hparams.num_hidden,
            hidden_size=hparams.num_hidden,
            num_layers=hparams.num_layers,
            batch_first=self.batch_first,
        )
        self.linear = torch.nn.Linear(hparams.num_hidden, num_labels)

    def forward(
        self, x: torch.Tensor, states: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        x, states = self.net(x, states)
        x = self.linear(x)
        return x, states


class CharGenerationSystem:
    def __init__(
        self, hparams: HyperParams,
    ):
        self.hparams = hparams
        self.data_splits = [Splits.train, Splits.val, Splits.test]
        self.device = get_device()
        self.datasets = {s: self.get_dataset(s) for s in self.data_splits}
        self.vocab_size = len(self.datasets[Splits.train].vocab)
        self.net = SequenceNet(
            num_vocab=self.vocab_size, num_labels=self.vocab_size, hparams=hparams,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=hparams.lr)

    def get_dataset(self, data_split: str):
        return StarTrekCharGenerationDataset(self.hparams, data_split)

    def get_loader(self, data_split: str) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.datasets[data_split],
            batch_size=self.hparams.bs,
            shuffle=(data_split == Splits.train),
        )

    def run_step(self, inputs, targets):
        outputs, states = self.net(inputs)
        outputs = outputs.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)
        loss = self.criterion(outputs, targets)

        predicts = torch.argmax(outputs.data, dim=1)
        acc = predicts.eq(targets).float().mean()
        return loss, acc

    def get_gradient_context(self, is_train: bool):
        if is_train:
            self.net.train()
            return torch.enable_grad
        else:
            self.net.eval()
            return torch.no_grad

    def run_epoch(self, data_split: str) -> Dict[str, float]:
        is_train = data_split == Splits.train
        acc_history = []
        loss_history = []
        steps_per_epoch = self.hparams.steps_per_epoch
        if data_split in {Splits.val, Splits.test}:
            steps_per_epoch = steps_per_epoch // 10

        with self.get_gradient_context(is_train)():
            while True:
                for inputs, targets in self.get_loader(data_split):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    if is_train:
                        self.optimizer.zero_grad()

                    loss, acc = self.run_step(inputs, targets)
                    acc_history.append(acc.item())
                    loss_history.append(loss.item())

                    if is_train:
                        loss.backward()
                        self.optimizer.step()

                    if self.hparams.dev_run or len(loss_history) > steps_per_epoch:
                        acc = np.round(np.mean(acc_history), decimals=3)
                        loss = np.round(np.mean(loss_history), decimals=3)
                        return dict(loss=loss, acc=acc)

    def train(self, early_stop=True) -> Tuple[List[dict], dict]:
        self.net.to(self.device)
        best_loss = 1e9
        best_weights = {}

        history = []
        for _ in tqdm(range(self.hparams.epochs)):
            results = {s: self.run_epoch(s) for s in self.data_splits}
            history.append(results)
            print(results)

            loss = results[Splits.val]["loss"]
            if loss < best_loss:
                best_loss = loss
                best_weights = copy.deepcopy(self.net.state_dict())
            elif early_stop:
                break
            if self.hparams.dev_run and len(history) >= 3:
                break

        return history, best_weights

    def sample(self, num: int = None, length: int = None):
        if num is None:
            num = self.hparams.bs
        if length is None:
            length = self.hparams.seq_len

        dataset: StarTrekCharGenerationDataset = self.datasets[Splits.train]
        token_start = dataset.vocab.stoi[dataset.vocab.start]
        x = torch.from_numpy(np.array([token_start] * num))
        x = x.long().reshape(num, 1)

        self.net.eval()
        sampler = dict(tcn=self.sample_tcn, lstm=self.sample_rnn, gru=self.sample_rnn)
        with torch.no_grad():
            outputs = sampler[self.hparams.model_name](x, length)

        for i in range(num):
            print(dataset.sequence_to_text(outputs[i, :]))

    def sample_rnn(self, x: torch.Tensor, length: int) -> torch.Tensor:
        states = None
        history = [x]
        for _ in range(length):
            logits, states = self.net(history[-1], states)
            # predicts = logits.argmax(dim=-1)
            predicts = torch.multinomial(
                torch.softmax(logits.squeeze(), dim=-1), num_samples=1
            )
            history.append(predicts)
        return torch.stack(history).squeeze().transpose(0, 1)

    def sample_tcn(self, x: torch.Tensor, length: int) -> torch.Tensor:
        states = None
        history = [x]
        for _ in range(length):
            inputs = torch.cat(history[-length:], dim=-1)
            logits, states = self.net(inputs, states)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            history.append(torch.multinomial(probs, num_samples=1))
        return torch.stack(history).squeeze().transpose(0, 1)


def main(dev_run=False, path_save_results="results.pt"):
    results = []
    for hparams in [
        # HyperParams(model_name="lstm", num_layers=1, dev_run=dev_run),  # val_loss=1.61
        # HyperParams(model_name="lstm", num_layers=2, dev_run=dev_run),  # val_loss=1.536
        HyperParams(model_name="tcn", num_layers=2, dev_run=dev_run),  # val_loss=1.643
    ]:
        system = CharGenerationSystem(hparams)
        history, weights = system.train()
        results.append(dict(hparams=hparams.__dict__, history=history, weights=weights))
    torch.save(results, path_save_results)


if __name__ == "__main__":
    fire.Fire(main)
