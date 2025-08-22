#!/usr/bin/env python3
import os, argparse, time
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.cuda.amp import GradScaler, autocast

# --- Random image dataset ---
class RandomImageDataset(Dataset):
    def __init__(self, n=20000, h=304, w=260, c=3, num_classes=10):
        self.data = torch.randn(n, c, h, w)
        self.targets = torch.randint(0, num_classes, (n,))
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

# --- Simple CNN ---
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=15*32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()

def setup(): dist.init_process_group(backend="nccl")
def cleanup(): dist.destroy_process_group()

def main():
    args = parse_args()
    setup()
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world      = int(os.environ["WORLD_SIZE"])
    device     = torch.device(f"cuda:{local_rank}")

    # Dataset + distributed sampler
    ds = RandomImageDataset(n=10000, h=304, w=260, c=3, num_classes=10)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.workers, pin_memory=True)

    # Model
    model = SmallCNN(num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    scaler= GradScaler()

    start = time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for X, y in dl:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                logits = model(X)
                loss = lossf(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        if rank == 0:
            print(f"Epoch {epoch+1}: loss={running/len(dl):.4f}")
    if rank == 0:
        print(f"Training finished in {time.time()-start:.2f}s")
    cleanup()

if __name__ == "__main__":
    main()


'''
Cmd to run it:

torchrun --standalone --nproc_per_node=2 train_ddp.py 
'''
