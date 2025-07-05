# Distributed Hetero ML

Train your machine learning models across multiple machines with heterogeneous hardware. Works with GPUs, CPUs, and Apple Silicon all at once.

## What this does

Sometimes you have a bunch of computers lying around with different specs. Maybe some have GPUs, some don't, some are Macs with M4 chips. This framework lets you use all of them together for training without much hassle.

It handles the coordination between machines, keeps track of gradients, and makes sure everyone stays in sync. Built on Ray so it's pretty solid.

## Basic usage

Here's how to train a simple model across your machines:

```python
import torch
import torch.nn as nn
from distributed_hetero_ml import DistributedTrainer, TrainingConfig

# Define your model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

# Create a model factory
class MyModelFactory:
    def create_model(self):
        return SimpleModel()
    
    def create_optimizer(self, model, config):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def create_criterion(self):
        return nn.CrossEntropyLoss()

# Create a data loader
class MyDataLoader:
    def get_batch(self, batch_size, device):
        # Your data loading logic here
        x = torch.randn(batch_size, 10).to(device)
        y = torch.randint(0, 2, (batch_size,)).to(device)
        return x, y
    
    def get_dataset_size(self):
        return 1000

# Set up training
config = TrainingConfig(
    batch_size=32,
    learning_rate=0.001,
    num_gpu_workers=2,
    num_cpu_workers=1
)

trainer = DistributedTrainer(
    model_factory=MyModelFactory(),
    data_loader=MyDataLoader(),
    config=config
)

# Train the model
results = trainer.train(num_iterations=100)
```

## Quick start for the lazy

If you just want to get something running quickly:

```python
from distributed_hetero_ml import quick_start

# This figures out your hardware and sets reasonable defaults
trainer = quick_start(
    model_factory=MyModelFactory(),
    data_loader=MyDataLoader(),
    num_iterations=50
)

results = trainer.train()
```

## Multi cluster setup

Connect to a Ray cluster running on other machines:

```python
trainer = DistributedTrainer(
    model_factory=MyModelFactory(),
    data_loader=MyDataLoader(),
    config=config,
    cluster_address="ray://head-node-ip:10001"
)
```

## Checkpointing

Save your progress so you don't lose everything:

```python
# Save a checkpoint
trainer.save_checkpoint("my_model_checkpoint.pt")

# Load it back later
trainer.load_checkpoint("my_model_checkpoint.pt")

# Or save automatically during training
trainer.train(
    num_iterations=100,
    checkpoint_path="auto_checkpoint"
)
```

## Context manager style

Clean up resources automatically:

```python
with DistributedTrainer(model_factory, data_loader, config) as trainer:
    results = trainer.train(num_iterations=100)
    trainer.save_checkpoint("final_model.pt")
```

## What hardware works

- **NVIDIA GPUs**: Uses CUDA, supports mixed precision training
- **Apple Silicon**: Uses MPS backend for Apple silicon chips
- **CPUs**: Falls back to CPU training when no accelerators available
- **Mixed setups**: Combines different hardware types in the same training run

The framework automatically detects what you have and configures itself accordingly.

## Configuration options

```python
config = TrainingConfig(
    batch_size=64,
    learning_rate=0.01,
    num_epochs=10,
    gradient_clipping=1.0,
    use_mixed_precision=True,
    checkpoint_interval=100,
    num_gpu_workers=2,
    num_cpu_workers=1,
    cpus_per_worker=2,
    gpus_per_worker=1
)
```

## Requirements

- Python 3.13+
- PyTorch 2.7+
- Ray 2.47+
- NumPy 2.3+

## Contributing

Found a bug? Have an idea? Open an issue or send a PR. The codebase is pretty straightforward to navigate.

## License

MIT License. Do whatever you want with it.
