from inference import SGANInference
from inference import computeDistances
from inference import get_model_matrix
import numpy as np
from sgan.data.loader import data_loader
import os
import torch

model_name = "eth_8_model"
model_path = "models/sgan-p-models/%s.pt" % model_name
trajPlan = SGANInference(model_path)
generator = trajPlan.generator
_args = trajPlan.args
_args.batch_size = 1
sets = [0]
_, loader = data_loader(_args, sets)
num_of_predictions = 5 #obs_len = 8
checkpoint = {
    'ground_truth': [],
    'mask':[],
    'future_mask': [],
    'model_matrix': [],
    'mse_metric': [],
    'interaction_time':[]
}
checkpoint_path = os.path.join(
    os.getcwd(), 'checkpoints', '%s_baseline_time_p.pt' % model_name
                )
print_every = 20
i = 0

for batch in loader:
    obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
    ground_truth, mask, render, seq_start_end = ground_truth_list[0], mask_list[0], render_list[0], seq_start_end[0] #batch_size = 1
    #print(ground_truth)
    future_mask = mask > 0
    if np.sum(future_mask) == 0:
        #print('not good batch')
        continue
    obs_traj = obs_traj.numpy()
    traj_length = len(obs_traj[1])
    matrices = []
    times = []
    predicted = obs_traj.transpose((1,0,2))
    for j in range(num_of_predictions):
        sub_matrices = np.zeros((len(predicted),len(predicted),2))
        closest_interaction = np.zeros((len(predicted),len(predicted)))
        history = predicted
        predicted = trajPlan.evaluate(history)
        for x in range(len(predicted)):
            cors, interaction_time = computeDistances(predicted,x, j) #adjancey matrix
            cors = np.array(cors)
            sub_matrices[x,:,:] = cors
            closest_interaction[x,:] = interaction_time
        #print('sub matrices')
        #print(sub_matrices)
        matrices.append(sub_matrices)
        times.append(closest_interaction)
    matrices = np.array(matrices)
    times = np.array(times)
    
    model_matrix, time = get_model_matrix(matrices, times)
    #print(model_matrix)
    mse_metric = np.sum(np.linalg.norm(model_matrix - ground_truth, axis = 2)*future_mask)
    #print(mse_metric)
    #IF MASK is
    checkpoint['ground_truth'].append(ground_truth)
    checkpoint['mask'].append(mask)
    checkpoint['future_mask'].append(future_mask)
    checkpoint['model_matrix'].append(model_matrix)
    checkpoint['mse_metric'].append(mse_metric)
    checkpoint['interaction_time'].append(time)
    torch.save(checkpoint, checkpoint_path)
    if (i % print_every == 0):
        print(mse_metric)
    i += 1




    
#print(trajPlan)

#take 1 batch
#make x number of predictions, find adjancey matrix and make a list of them
#find the best matrix out of them all (minimum)