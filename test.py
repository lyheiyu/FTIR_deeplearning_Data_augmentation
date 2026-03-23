import wandb
import random

wandb.init(name='test3', project="test1")

i=0
for i in range(50):
    y = random.randint(0,10)
    wandb.log({'Y': y})
    print("ep ", i)