'''
Data parameters.

    `NUM_FEATURES`: int = 50
        Number of features, aka input data dimensionality

    `NOISE_LEVEL`: float
        Scales the noise added to the data

    `N_CHUNKS`: int = 50
        Number of chunks to divide dataset

    `NUM_SAMPLES`: int = 50
        Number of samples

    REPORTED_ARI: list[int]
        ARI scores reported in original paper
    
    `CHUNK_SIZE`: int = 50
        Chunk size to split up dataset
'''

N_FEATURES: int = 50
'''Number of features, aka input data dimensionality'''

NOISE_LEVEL: float = 0.1
'''Scales the noise added to the data'''

N_CHUNKS: int = 3
'''Number of chunks to divide dataset'''

N_SAMPLES: int = 500
'''Number of samples'''

REPORTED_ARI: list[int] = [0.524,0.455,0.280]
'''ARI scores reported in original paper'''

CHUNK_SIZE: int = N_SAMPLES // N_CHUNKS
'''Chunk size to split up dataset'''
