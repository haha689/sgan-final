from inference import SGANInference
from inference import computeDistances
from inference import get_model_matrix
import numpy as np
from sgan.data.loader import data_loader

trajPlan = SGANInference("models/sgan-models/eth_8_model.pt")
generator = trajPlan.generator
_args = trajPlan.args
_args.batch_size = 1
sets = [0]
_, loader = data_loader(_args, sets)
num_of_predictions = 50

for batch in loader:
    obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
    ground_truth, mask, render, seq_start_end = ground_truth_list[0], mask_list[0], render_list[0], seq_start_end[0] #batch_size = 1
    print(ground_truth)
    if np.sum(mask) == 0:
        print('not good batch')
        break
    obs_traj = obs_traj.numpy()
    traj_length = len(obs_traj[1])
    matrices = []
    predicted = obs_traj.transpose((1,0,2))
    for j in range(num_of_predictions):
        sub_matrices = np.zeros((len(predicted),len(predicted),2))
        history = predicted
        predicted = trajPlan.evaluate(history)
        for x in range(len(predicted)):
            cors = computeDistances(predicted,x) #adjancey matrix
            cors = np.array(cors)
            sub_matrices[x,:,:] = cors
        #print('sub matrices')
        #print(sub_matrices)
        matrices.append(sub_matrices)
    matrices = np.array(matrices)
    
    model_matrix = get_model_matrix(matrices)
    #print(model_matrix)
    mse_metric = np.sum(np.linalg.norm(model_matrix - ground_truth, axis = 2)*mask)
    #print(mse_metric)
    #IF MASK is
    break





    
#print(trajPlan)

#take 1 batch
#make x number of predictions, find adjancey matrix and make a list of them
#find the best matrix out of them all (minimum)
