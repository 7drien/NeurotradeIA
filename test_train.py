import queue
from src.train import train_model_with_callback
q = queue.Queue()
# To prevent it from training 50 epochs, we'll just modify config temporarily or mock it.
# Actually let's just run it for 1 epoch.
import src.config
src.config.EPOCHS = 1
train_model_with_callback(q)
while not q.empty():
    print(q.get())
