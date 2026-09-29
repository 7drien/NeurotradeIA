import queue
import sys

# Patch config before anything imports it
import src.config
src.config.EPOCHS = 1

from src.train import train_model_with_callback

q = queue.Queue()
train_model_with_callback(q)

while not q.empty():
    print(q.get())
