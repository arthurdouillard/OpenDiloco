import os
from contextlib import nullcontext

import torch
from pydantic_config import parse_argv, validate_call
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)

from hivemind.optim.optimizer import logger


def ddp_setup():
    init_process_group(backend="gloo")


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


@validate_call
def train(
    lr: float = 4e-4, total_batch_size: int = 512, per_device_train_batch_size: int = 32, max_steps: int | None = None
):
    _local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _rank = int(os.environ["RANK"])

    assert total_batch_size % world_size == 0
    batch_size = total_batch_size // world_size

    assert batch_size % per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    INPUT_DIM = 100
    OUTPUT_DIM = 100

    model = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM)

    model = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD, use_orig_params=True, device_id="cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    model.train()
    loss_batch = 0

    for step in range(10000):
        real_step = (step + 1) // gradient_accumulation_steps
        is_accumulating = bool((step + 1) % gradient_accumulation_steps)

        input = torch.rand(per_device_train_batch_size, INPUT_DIM)
        target = torch.randint(0, OUTPUT_DIM, (per_device_train_batch_size,))

        with model.no_sync() if is_accumulating else nullcontext():
            output = model(input)

            loss = torch.nn.functional.cross_entropy(output, target)
            loss = loss / gradient_accumulation_steps
            loss_batch += loss.detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            log(f"step: {real_step}, loss: {loss_batch.item()}")

            loss_batch = 0

            if max_steps is not None and real_step >= max_steps:
                break

    log("Training completed.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    train(**parse_argv())
    destroy_process_group()
