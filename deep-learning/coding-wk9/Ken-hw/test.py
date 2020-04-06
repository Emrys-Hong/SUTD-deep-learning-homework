import torch
from train import CharGenerationSystem, HyperParams, get_device


def main(path_save_results="results.pt"):
    results = torch.load(path_save_results, map_location=get_device())
    for r in results:
        system = CharGenerationSystem(HyperParams(**r["hparams"]))
        system.net.load_state_dict(r["weights"])
        for h in r["history"]:
            print(h)
        system.sample(length=100)


if __name__ == "__main__":
    main()
