import torch
import numpy as np
import pandas as pd
from general_training import eulerStepNumpy_HarmOsc, eulerStepNumpy_Kepler, eulerStepNumpy_Pendulum, verletStep_HarmOsc, verletStep_Kepler, verletStep_Pendulum, verletStepNumpy_HarmOsc, verletStepNumpy_Kepler, verletStepNumpy_Pendulum
from TrainingData.general_problems import HarmOsc, Pendulum, Kepler
from TrainingData import verlet8
import itertools
import copy
import multiprocessing
from functools import partial

'''
method, problem, sym, sch, linear, best, N_train, M_test, tau_txt, eta1_txt, eta2_txt, nL, nN, nM
training_type = (method, problem, sym, sch, linear, best, learning_rate, eta1_txt, eta2_txt)
training_params = (nL, nN, nM, N_train, M_test, tau_txt)
'''

def load_model(method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau, learning_rate, eta1, eta2, nL, nN, nM, load_only_model = True):
    tau_txt = str(tau).replace('.', '')
    epochs_th = str(epochs/1000).replace('.0', '')
    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')


    # Load selected model
    if not sym:
        if sch:
            model_name = f"TrainedModels/{method}/{problem}/sch{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
        else:
            model_name = f"TrainedModels/{method}/{problem}/{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
                
        if linear:
            model_name = model_name.replace(f'{problem}Rand', f'{problem}LinearRand')

        if best:
            model_name = model_name.replace('TH', 'TH_best')

    if sym:
        if sch:
            model_name = f"TrainedModels/{method}/Sym/{problem}/sch{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
        else:
            model_name = f"TrainedModels/{method}/Sym/{problem}/{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
                
        if linear:
            model_name = model_name.replace(f'{problem}Rand', f'{problem}LinearRand')

        if best:
            model_name = model_name.replace('TH', 'TH_best')

    if load_only_model:
        model, *_ = torch.load(model_name)
        return model
    else:
        model, loss, acc, *_ = torch.load(model_name)
        return model, loss, acc
    
def loss_acc_average(loss_acc_params):
    method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau, learning_rate, eta1, eta2, nL, nN, nM, load_only_model = loss_acc_params
    loss_averages = []
    loss_stds = []
    acc_averages = []
    acc_stds = []
    nLs = []
    nNs = []

    for layer in nL:
        for width in nN:
            nLs.append(layer)
            nNs.append(width)
            losses = []
            accs = []

            for run in nM:
                _, loss, acc = load_model(method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau, learning_rate, eta1, eta2, layer, width, run, load_only_model)
                losses.append(loss)
                accs.append(acc)

            losses_min = []
            accs_min = []
            for i in range(len(nM)):
                if not best:
                    losses_min.append(losses[i][-1])
                    accs_min.append(accs[i][-1])
                else:
                    min_index = np.argmin(accs[i])
                    losses_min.append(losses[i][min_index])
                    accs_min.append(accs[i][min_index])


            loss_avg = np.average(losses_min)
            loss_std = np.std(losses_min)
            acc_avg = np.average(accs_min)
            acc_std = np.std(accs_min)

            loss_averages.append(loss_avg)
            loss_stds.append(loss_std)
            acc_averages.append(acc_avg)
            acc_stds.append(acc_std)

    df1 = pd.DataFrame(
        {
            "nL": nLs,
            "nN": nNs,
            "Loss average": loss_averages,
            "Loss std": loss_stds,
            "Acc average": acc_averages,
            "Acc std": acc_stds,
        }
    )

    return loss_averages, loss_stds, acc_averages, acc_stds, df1

def method_problem_select(method, problem):
    if method == 'Euler':
        if problem == 'HarmOsc':
            numeric_stepNumpy = eulerStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = eulerStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = eulerStepNumpy_Kepler
    elif method == "Verlet":
        if problem == 'HarmOsc':
            numeric_stepNumpy = verletStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = verletStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = verletStepNumpy_Kepler

    if problem == 'HarmOsc':
        probl_class = HarmOsc
    elif problem == 'Pendulum':
        probl_class = Pendulum
    elif problem == 'Kepler':
        probl_class = Kepler

    return numeric_stepNumpy, probl_class

def model_predict_Tend(model, sym, x0, tau, Tend, numeric_stepNumpy, extraParams):
    device = 'cpu'
    d = len(x0)
    M = int(Tend/tau)

    pred_inv = np.zeros([M+1, d])
    Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
    if sym:
        Tau = torch.tensor([[[tau**2]]], dtype=torch.float64, device=device)
    else:
        Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)

    with torch.no_grad():
        inverse, _ = model.back(Z, Tau) # Pass trough inverse model

    pred_inv[0] = inverse.reshape((1, d)).numpy()

    for i in range(M):
        pred_inv[i+1] = numeric_stepNumpy(pred_inv[i], tau, extraParams)
        
    pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape((M+1, 1, d))
    with torch.no_grad():
        pred, _ = model(pred_inv, Tau) # Pass trough original model

    pred = pred.numpy().reshape(M+1, d)

    return pred

def numeric_predict_Tend(x0, tau, Tend, numeric_stepNumpy, extraParams):
    d = len(x0)
    M = int(Tend/tau)

    pred_numeric = np.zeros([M+1, d])
    pred_numeric[0, :] = x0

    for i in range(M): 
        pred_numeric[i+1] = numeric_stepNumpy(pred_numeric[i], tau, extraParams)

    return pred_numeric

def analytical_pred_Tend(x0, tau, Tend, probl_class, extraParams):
    d = len(x0)
    D = int(d/2)
    M = int(Tend/tau)

    exact = np.zeros([M+1, d])
    exact[0] = x0
    for i in range(M):
        exact[i+1] = verlet8.eight_orderPrecise(exact[i], tau, D, probl_class, extraParams)

    return exact

def generate_starting_points(area, seed, nr_trajects):
    D = len(area[0]) # Half dimension of problem
    x0 = []

    q_transform = (area[0, :, 1] -area[0, :, 0])
    p_transform = (area[1, :, 1] -area[1, :, 0])
    np.random.seed(seed)

    for i in range(nr_trajects):
        x0.append([])
        for j in range(D): # Append all q values
            x0[-1].append(q_transform[j]*np.random.rand() +area[0, :, 0][0])
        
        for j in range(D): # Append all p values
            x0[-1].append(p_transform[j]*np.random.rand() +area[1, :, 0][0])

    return x0

def generate_starting_pointsKepler(seed, nr_trajects):
    np.random.seed(seed)
    x0 = []

    for i in range(nr_trajects):
        x0.append([])
        e = np.random.rand()*0.1 +0.55 # in range [0.1, 0.9]
        x0[-1].append(1-e)
        x0[-1].append(0)
        x0[-1].append(0)
        p2 = np.sqrt((1+e)/(1-e))
        x0[-1].append(p2)

    return x0

def error_calc(problem, M, d, exacts, predictions, nr_trajects, probl_class, extraParams):
    Err = np.zeros([M+1, 1])
    Err_Tend = []
    HErr = np.zeros([M+1, 1])
    HErr_max = []


    for k in range(nr_trajects):
        Err_tmp = np.sqrt(np.sum((predictions[k] -exacts[k])**2, 1)).reshape((M+1, 1))
        Err_Tend.append(Err_tmp[-1])
        Err += Err_tmp

        H0 = probl_class.H(exacts[k][0, :].reshape((1, d)), extraParams)
        HErr_tmp = np.abs((probl_class.H(predictions[k], extraParams).reshape((M+1, 1)) -H0)/H0)
        HErr_max.append(max(HErr_tmp))
        HErr += HErr_tmp

    Err = Err/nr_trajects
    Err_std = np.std(Err_Tend)
    HErr = HErr/nr_trajects
    HErr_std = np.std(HErr_max)

    if problem == "Kepler":
        LErr = np.zeros([M+1, 1])
        LErr_max = []
        for k in range(nr_trajects):
            L0 = probl_class.L(exacts[k][0, 0:d].reshape((1, d)), extraParams)
            LErr_tmp = np.abs((probl_class.L(predictions[k], extraParams).reshape((M+1, 1)) -L0)/L0)
            LErr_max.append(max(LErr_tmp))
            LErr += LErr_tmp

        LErr = LErr/nr_trajects
        LErr_std = np.std(LErr_max)

    if problem == "Kepler":
        return Err, Err_std, HErr, HErr_std, LErr, LErr_std
    else:
        return Err, Err_std, HErr, HErr_std

def errors_average(loss_acc_params, error_params, tau):
    method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau_model, learning_rate, eta1, eta2, nL, nN, nM, load_only_model = loss_acc_params
    area, nr_trajects, seed, Tend, extraParams = error_params

    if problem == "Kepler":
        x0 = generate_starting_pointsKepler(seed, nr_trajects)
    else:
        x0 = generate_starting_points(area, seed, nr_trajects)
        
    numeric_stepNumpy, probl_class = method_problem_select(method, problem)
    
    ### Get kernel predictions
    preds_numeric = []
    for i in range(nr_trajects):
        pred_numeric = numeric_predict_Tend(x0[i], tau, Tend, numeric_stepNumpy, extraParams)
        preds_numeric.append(pred_numeric)

    ### Get "analytical" prediction or ground truth
    exacts = []
    for i in range(nr_trajects):
        exact = analytical_pred_Tend(x0[i], tau, Tend, probl_class, extraParams)
        exacts.append(exact)

    ### Get some variables of dimesnion and step count
    D = len(area[0]) # Half dimension of problem
    d = 2*D
    M = int(Tend/tau)

    ### Calculate kernel errors
    if problem == "Kepler":
        Err_numeric, Err_numeric_std, HErr_numeric, HErr_numeric_std, LErr_numeric, LErr_numeric_std = error_calc(problem, M, d, exacts, preds_numeric, nr_trajects, probl_class, extraParams)
    else:
        Err_numeric, Err_numeric_std , HErr_numeric, HErr_numeric_std = error_calc(problem, M, d, exacts, preds_numeric, nr_trajects, probl_class, extraParams)
    
    ### 
    if problem == "Kepler":
        df_kernel = pd.DataFrame(
            {
                "Avg Err at Tend": list(itertools.chain(*[Err_numeric[-1]])),
                "Std1": [Err_numeric_std],
                "Avg max abs HErr": [np.max(HErr_numeric)],
                "Std2": [HErr_numeric_std],
                "Avg max abs LErr": [np.max(LErr_numeric)],
                "Std3": [LErr_numeric_std],
            }
        )
    else:
        df_kernel = pd.DataFrame(
            {
                "Avg Err at Tend": list(itertools.chain(*[Err_numeric[-1]])),
                "Std1": [Err_numeric_std],
                "Avg max abs HErr": [np.max(HErr_numeric)],
                "Std2": [HErr_numeric_std],
            }
        )

    ### Get all model predictions
    nLs = []
    nNs = []
    Err_preds_Tend = [] # Pred avg for every model
    Err_stds_all = []
    HErr_preds_max = []
    HErr_stds_all = []
    if problem == "Kepler":
        LErr_preds_max = []
        LErr_stds_all = []
    
    df_pred_individuals = [] 

    for layer in nL:
        for width in nN:
            nLs.append(layer)
            nNs.append(width)

            Err_pred_Tend = [] # Pred avg for individual model
            HErr_pred_max = []
            if problem == "Kepler":
                LErr_pred_max = []

            for run in nM:
                model = load_model(method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau_model, learning_rate, eta1, eta2, layer, width, run, load_only_model)
                
                Err_pred = np.zeros([M+1, 1]) # Errors for this specific model for all runs
                HErr_pred = np.zeros([M+1, 1])
                if problem == "Kepler":
                    LErr_pred = np.zeros([M+1, 1])

                preds = []
                Err_stds = []
                
                HErr_stds = []
                
                if problem == "Kepler":
                    LErr_stds = []


                for i in range(nr_trajects):
                    pred = model_predict_Tend(model, sym, x0[i], tau, Tend, numeric_stepNumpy, extraParams)
                    preds.append(pred)

                if problem == "Kepler":
                    Err, Err_std, HErr, HErr_std, LErr, LErr_std = error_calc(problem, M, d, exacts, preds, nr_trajects, probl_class, extraParams)
                    Err_stds.append(Err_std)
                    HErr_stds.append(HErr_std)
                    LErr_stds.append(LErr_std)
                    Err_pred_Tend.append(Err[-1])
                    HErr_pred_max.append(np.max(HErr))
                    LErr_pred_max.append(np.max(LErr))
                    Err_pred += Err
                    HErr_pred += HErr
                    LErr_pred += LErr
                else:
                    Err, Err_std, HErr, HErr_std = error_calc(problem, M, d, exacts, preds, nr_trajects, probl_class, extraParams)
                    Err_stds.append(Err_std)
                    HErr_stds.append(HErr_std)
                    Err_pred_Tend.append(Err[-1])
                    HErr_pred_max.append(np.max(HErr))
                    Err_pred += Err
                    HErr_pred += HErr
            
            #return len(nM)*[nL], len(nM)*[nN], nM, Err_pred_Tend, HErr_pred_max
            ### Averages for every model in run to put into dataframe and later select best
            if problem == "Kepler":
                df_pred_individual = pd.DataFrame(
                    {
                        "nL": len(nM)*[layer],
                        "nN": len(nM)*[width],
                        "nM": nM,
                        "Avg Err at Tend": list(itertools.chain(*Err_pred_Tend)),
                        "Avg max abs HErr": HErr_pred_max,
                        "Abg max abs LErr": LErr_pred_max,
                    }
                )
            else:
                df_pred_individual = pd.DataFrame(
                    {
                        "nL": len(nM)*[layer],
                        "nN": len(nM)*[width],
                        "nM": nM,
                        "Avg Err at Tend": list(itertools.chain(*Err_pred_Tend)),
                        "Avg max abs HErr": HErr_pred_max,
                    }
                )
            
            df_pred_individuals.append(df_pred_individual)

            ### Averages, and stds for whole specific model run
            Err_pred += Err_pred/len(nM)
            Err_preds_Tend.append(Err_pred[-1])
            Err_stds_all.append(np.mean(Err_stds))
            HErr_pred += HErr_pred/len(nM)
            HErr_preds_max.append(np.max(HErr_pred))
            HErr_stds_all.append(np.mean(HErr_stds))
            if problem == "Kepler":
                LErr_pred += LErr_pred/len(nM)
                LErr_preds_max.append(np.max(LErr_pred))
                LErr_stds_all = np.mean(LErr_stds)


    if problem == "Kepler":
        df_pred = pd.DataFrame(
            {
                "nL": nLs,
                "nN": nNs,
                "Avg Err at Tend": list(itertools.chain(*Err_preds_Tend)),
                "Std1": Err_stds_all,
                "Avg max abs HErr": HErr_preds_max,
                "Std2": HErr_stds_all,
                "Abg max abs LErr": LErr_preds_max,
                "Std3": LErr_stds_all,
            }
        )
    else:
        df_pred = pd.DataFrame(
            {
                "nL": nLs,
                "nN": nNs,
                "Avg Err at Tend": list(itertools.chain(*Err_preds_Tend)),
                "Std1": Err_stds_all,
                "Avg max abs HErr": HErr_preds_max,
                "Std2": HErr_stds_all,
            }
        )

    return df_kernel, df_pred, df_pred_individuals

def model_predict_oneStep(model, sym, x0, tau, numeric_stepNumpy, extraParams):
    device = 'cpu'
    d = len(x0)

    Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
    if sym:
        Tau = torch.tensor([[[tau**2]]], dtype=torch.float64, device=device)
    else:
        Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)

    with torch.no_grad():
        inverse, _ = model.back(Z, Tau) # Pass trough inverse model

    inverse = inverse.reshape((1, d)).numpy()
    pred_inv = numeric_stepNumpy(inverse, tau, extraParams)
        
    pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape((1, 1, d))
    with torch.no_grad():
        pred, _ = model(pred_inv, Tau) # Pass trough original model

    pred = pred.numpy().reshape(1, d)

    return pred

def numeric_predict_oneStep(x0, tau, numeric_stepNumpy, extraParams):
    d = len(x0)
    pred_numeric = numeric_stepNumpy(x0, tau, extraParams)

    return pred_numeric

def analytical_pred_oneStep(x0, tau, probl_class, extraParams):
    d = len(x0)
    D = int(d/2)

    exact = verlet8.eight_orderPrecise(x0, tau, D, probl_class, extraParams)

    return exact

def rmnse(exact, pred):
    square_error = np.square(exact -pred)
    mse = np.mean(square_error, axis=1)
    root_mse = np.sqrt(mse)
    std = np.std(exact, axis=0)
    normalized_square_error = square_error/np.square(std)
    mean_normalized_square_error = np.mean(normalized_square_error, axis = 1)
    root_mean_normalized_square_error = np.sqrt(mean_normalized_square_error)

    return root_mean_normalized_square_error

def vpt(exact, pred, tau, treshold):
    
    rmnse_pred = rmnse(exact, pred)
    M = len(rmnse_pred)
    VPT = None
    for i in range(M):
        if rmnse_pred[i] > treshold:
            VPT = tau*i
            break

    return VPT

def expand_array(arr, new_shape): # Because why would numpy have this built in
    expanded_arr = np.zeros(new_shape, dtype=arr.dtype)
    expanded_arr[:arr.shape[0], :arr.shape[1]] = arr
    return expanded_arr

def analytical_long(loss_acc_params, error_params, Tend, tau):

    method, problem, sym, sch, linear, best, epochs, N_train, M_test, _, learning_rate, eta1, eta2, nL, nN, nM, load_only_model = loss_acc_params
    area, nr_trajects, seed, _, extraParams = error_params

    if problem == "Kepler":
        x0 = generate_starting_pointsKepler(seed, nr_trajects)
    else:
        x0 = generate_starting_points(area, seed, nr_trajects)

    numeric_stepNumpy, probl_class = method_problem_select(method, problem)
    tau_txt = str(tau).replace('.', '')

    ## Get "analytical" prediction or ground truth
    exacts = []
    for i in range(nr_trajects):
        exact = analytical_pred_Tend(x0[i], tau, Tend, probl_class, extraParams)
        exacts.append(exact)
    
    np.savez(f'Analysis/VPT/{problem}Const{tau_txt}Tau_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend)}_analytical', exacts)
    return 'analytical done'
        
def kernel_long(loss_acc_params, error_params, Tend, tau):
    method, problem, sym, sch, linear, best, epochs, N_train, M_test, _, learning_rate, eta1, eta2, nL, nN, nM, load_only_model = loss_acc_params
    area, nr_trajects, seed, _, extraParams = error_params

    if problem == "Kepler":
        x0 = generate_starting_pointsKepler(seed, nr_trajects)
    else:
        x0 = generate_starting_points(area, seed, nr_trajects)

    numeric_stepNumpy, probl_class = method_problem_select(method, problem)
    tau_txt = str(tau).replace('.', '')


    # Get kernel predictions
    kernels = []
    for i in range(nr_trajects):
        kernel = numeric_predict_Tend(x0[i], tau, Tend, numeric_stepNumpy, extraParams)
        kernels.append(kernel)
    
    np.savez(f'Analysis/VPT/{problem}Const{tau_txt}Tau_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend)}_{method}', kernels)
    return 'kernel done'

def pred_long(loss_acc_params, error_params, Tend, tau):
    method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau_model, learning_rate, eta1, eta2, nL, nN, nM, load_only_model = loss_acc_params
    area, nr_trajects, seed, _, extraParams = error_params

    if problem == "Kepler":
        x0 = generate_starting_pointsKepler(seed, nr_trajects)
    else:
        x0 = generate_starting_points(area, seed, nr_trajects)
        
    numeric_stepNumpy, probl_class = method_problem_select(method, problem)
    tau_txt = str(tau).replace('.', '')
    tau_model_txt = str(tau_model).replace('.', '')

    pred_all = []
    for i in nM:
        model = load_model(method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau_model, learning_rate, eta1, eta2, nL, nN, i, load_only_model)
        # Get model predictions
        preds = []
        for i in range(nr_trajects):
            pred = model_predict_Tend(model, sym, x0[i], tau, Tend, numeric_stepNumpy, extraParams)
            preds.append(pred)

        pred_all.append(preds)

    epochs_th = str(epochs/1000).replace('.0', '')
    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

    name = f'Analysis/VPT/{problem}Const{tau_txt}Tau_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend)}'
    if sch:
        name += '_sch'
    
    name += f'_Const{tau_model_txt}Tau{epochs_th}TH'
    if best:
        name += '_best'

    if sch:
        name += f'_{eta1_txt}eta1_{eta2_txt}eta2'
    else:
        name += f'_{eta1_txt}eta1'

    name += f'_{nL}L{nN}n'
    

    np.savez(name, pred_all)
    return 'model done'

def vpt_calc(analytical, kernel, model_pred, nM, tau, Tend, treshold, nr_trajects, method):
    vpt_kernels = []
    for i in range(nr_trajects):
        vpt_kernel = vpt(analytical[i], kernel[i], tau, treshold)
        if vpt_kernel == None: # Assign max, if doesn't reach treshold at all
            vpt_kernel = Tend

        vpt_kernels.append(vpt_kernel)

    vpt_models = []
    for j in nM:
        for i in range(nr_trajects):
            vpt_model = vpt(analytical[i], model_pred[j][i], tau, treshold)
            if vpt_model == None: # Assign max, if doesn't reach treshold at all
                vpt_model = Tend
            vpt_models.append(vpt_model)


    vpt_kernel_avg = np.average(vpt_kernels)
    vpt_model_avg = np.average(vpt_models)

    df1 = pd.DataFrame(
        {
            "Tau": [tau],
            "Kernel": [method],
            "Avg VPT": [vpt_kernel_avg],
        }
    )

    df2 = pd.DataFrame(
        {
            "Tau": [tau],
            "Kernel:": [method],
            "Avg VPT": [vpt_model_avg],
        }
    )

    df3 = pd.DataFrame(
        {
            "Method:": [method, f"processing with {method}"],
            f"Avg VPT with tau={str(tau)}": [vpt_kernel_avg, vpt_model_avg],
        }
    )

    return df1, df2, df3, vpt_kernels, vpt_models  

