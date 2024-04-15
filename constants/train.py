'''
Training parameters.

    `BATCH_SIZE`: int
        Training batchsize

    `SKIP_NOISY_FULL`: bool
        Skip training noisy dataset after first iteration
    
    `EPOCHS`: int
        Number of epochs for training
    
    `LEARNING_RATE`: bool
        Learning rate for optimizer
    
    `RUNS`: int
        Total number of runs
    
    `SHUFFLE`: bool
        Shuffles the dataset
'''


BATCH_SIZE: int = 32
'''Training batchsize'''

SKIP_NOISY_FULL: bool = True
'''Skip training noisy dataset after first iteration'''

EPOCHS: int = 300
'''Number of epochs for training'''

LR: float = 0.005
'''Learning rate for optimizer'''

RUNS: int = 10
'''Total number of runs'''

SHUFFLE: bool = True
'''Shuffles the dataset'''
