import numpy as np
from sklearn.metrics import mean_squared_error 

def acc_balanced(pred, orig):
    idx_1, idx_0 = np.where(np.array(orig)==1)[0], np.where(np.array(orig)==0)[0]
    acc_1, acc_0 = np.mean(np.array(pred[idx_1])==1), np.mean(np.array(pred[idx_0])==0)
    print(f"Acc 1: {acc_1}, Acc 0: {acc_0}, Acc Avg: {np.mean([acc_1,acc_0])}")
    
def mse_balanced(pred, orig, r):
    orig = np.array(orig)

    idx_1, idx_0 = np.where(np.array(r)==1)[0], np.where(np.array(r)==0)[0]
 
    mse_1 = mean_squared_error(pred[idx_1], orig[idx_1])
    mse_0 = mean_squared_error(pred[idx_0], orig[idx_0])
    dev_1, var_1 = np.mean(pred[idx_1]-orig[idx_1]), np.var(pred[idx_1]-orig[idx_1])
    dev_0, var_0 = np.mean(pred[idx_0]-orig[idx_0]), np.var(pred[idx_0]-orig[idx_0])
    
    print(f"MSE 1: {mse_1}, Acc 0: {mse_0}, Acc Avg: {np.mean([mse_1,mse_0])}")
    print(f"Avg. Deviation 1: {dev_1} ± {np.sqrt(var_1)}")
    print(f"Avg. Deviation 0: {dev_0} ± {np.sqrt(var_0)}")