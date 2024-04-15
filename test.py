'''
Generates ARI scores for clustering via BasisDeVAE.

Christopher Korfmann 214474670
'''

from constants import data, env, hyperparams, train
# from synth_data_gen import generate_data
from data_generation import generate_data, plot_examples
from utils import add_noise, copy_data_into_tensors, generate_dataloaders, get_scores, save_boxplot, print_stats, save_scores, train_clusterings_on_datasets


# global model params shared by encoder and decoder
SHARED_PARAMS = {'num_features'   : data.NUM_FEATURES,
                 'device'     : env.DEVICE,
                 'linear_dims' : hyperparams.LINEAR_DIMS,
                 'latent_dims'      : hyperparams.LATENT_DIMS}

# global encoder specific params
ENCODER_PARAMS = SHARED_PARAMS

# global decoder specific params
DECODER_PARAMS = {'num_basis' : hyperparams.NUM_BASIS,
                  **SHARED_PARAMS}

# global VAE params
VAE_PARAMS = {'device' : env.DEVICE,
              'learning_rate'     : train.LEARNING_RATE}

# aggregate all model params
MODEL_PARAMS = {'encoder' : ENCODER_PARAMS,
                'decoder' : DECODER_PARAMS,
                'vae'     : VAE_PARAMS}



if __name__ == '__main__':
    # this dict will hold scores across runs
    global_scores: dict[str, list[float]] = {'gt_lo' : list(),
                                             'gt_hi' : list(),
                                             'lo_hi' : list()}

    for i in range(train.RUNS):
        print(f'{"-"*30}', f'Iter {i}', sep='\n')

        original_data = generate_data(num_samples=data.NUM_SAMPLES,
                                      num_features=data.NUM_FEATURES,
                                      dtype=env.NUMPY_DTYPE,
                                      fname='verbatim.csv',
                                      save_to_file=False if i <= 0 else False) # save data to csv on first run
        
        if i <= 0:
            plot_examples([ original_data[ : , : 30 ],      # gaussian features
                            original_data[ : , 30 : 40 ],   # monotonically increasing features
                            original_data[ : , 40 : ] ],    # monotonically decreasing features
                          ['red', 'green', 'blue'], # colors to match original paper
                          save=True)

        data_noisy = add_noise(original_data, data.NOISE_LEVEL, env.NUMPY_DTYPE)
        
        # aggregate data
        data_dict = {'gt' : original_data,
                     'hi' : data_noisy[ data.CHUNK_SIZE : , : ],     # high z
                     'lo' : data_noisy[ : -data.CHUNK_SIZE , : ]}    # low z
        
        if i <= 0 or not train.SKIP_NOISY_FULL:
            data_dict['noisy'] = data_noisy
        
        # make data Pytorch friendly
        datasets = copy_data_into_tensors(data_dict, env.DEVICE, env.TORCH_DTYPE)
        dataloaders = generate_dataloaders(datasets, train.BATCH_SIZE)

        # get clustering assignments
        clustering_assignments = train_clusterings_on_datasets(dataloaders,
                                                               model_params=MODEL_PARAMS,
                                                               dtype=env.TORCH_DTYPE,
                                                               epochs=train.EPOCHS,
                                                               save=True if (i <= 0 or not train.SKIP_NOISY_FULL) else False, # save state_dict for first run
                                                               save_dir=env.MODEL_DIR)
        
        # calculate ARI score and add to the global list
        scores_ = get_scores(clustering_assignments, ['lo','hi'], gt='gt')
        for k, v in global_scores.items():
            v.append(scores_[k])

    save_scores(global_scores, save_dir=env.RESULTS_DIR, fname=f'scores.csv')
    save_boxplot(global_scores, fname='boxplot', save_dir=env.RESULTS_DIR, usermedians=data.REPORTED_ARI)
    print_stats(global_scores)
    
