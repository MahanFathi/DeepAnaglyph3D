

def train(cfg, optimizer, dataset):

    while True:
        for iteration, (x, y) in enumerate(dataset):
            loss = optimizer.step(x, y)
            print(loss)
