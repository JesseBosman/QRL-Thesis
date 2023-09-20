import numpy as np

def entropy_loss(logits):
    logits = np.array(logits)
    return np.sum(logits*np.log(logits))

print(entropy_loss([0.97, 0.01, 0.01, 0.01]))