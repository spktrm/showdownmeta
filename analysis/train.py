import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

from analysis.dataset import TrainDataset
from analysis.model import Model


def train(
    model: nn.Module,
    device: torch.device,
    dataset: TrainDataset,
    optimizer: optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
):
    model.train()

    # ce_loss = nn.CrossEntropyLoss()
    # bce_loss = nn.BCEWithLogitsLoss()
    # mse_loss = nn.MSELoss()

    triplet_loss = nn.TripletMarginLoss()

    for n in range(1000):
        anc, pos, neg = dataset.get_batch(args.batch_size)

        anc = anc.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        neg = neg.to(device, non_blocking=True)

        optimizer.zero_grad()

        anc = model(anc)
        pos = model(pos)
        neg = model(neg)

        # (
        #     species_pred,
        #     ability_pred,
        #     item_pred,
        #     moves_pred,
        #     nature_pred,
        #     evs_pred,
        # ) = pred

        # (
        #     species_target,
        #     ability_target,
        #     item_target,
        #     moves_target,
        #     nature_target,
        #     evs_target,
        # ) = target

        # loss = ce_loss(species_pred.flatten(0, 1), species_target.flatten(0, 1))
        # loss += ce_loss(ability_pred.flatten(0, 1), ability_target.flatten(0, 1))
        # loss += ce_loss(item_pred.flatten(0, 1), item_target.flatten(0, 1))
        # loss += bce_loss(
        #     moves_pred.flatten(0, 1),
        #     F.one_hot(moves_target.flatten(0, 1), moves_pred.shape[-1]).sum(-2).float(),
        # )
        # loss += ce_loss(nature_pred.flatten(0, 1), nature_target.flatten(0, 1))
        # loss += mse_loss(evs_pred.flatten(0, 1), evs_target.flatten(0, 1))

        loss = triplet_loss(anc, pos, neg)

        loss.backward()
        optimizer.step()

        print(f"Step: {n} \tLoss: {loss.item():.6f}")

        if n % 100 == 0:
            print("saving...")
            torch.save(model.state_dict(), "weights.pt")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace the following lines with your dataset and model
    train_dataset = TrainDataset(
        "https://github.com/pkmn/smogon/raw/main/data/stats/gen9ou.json"
    )

    model = Model(9, 512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, device, train_dataset, optimizer, 1, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch Training Loop")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="learning rate (default: 0.01)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
