import os

current_dir = os.getcwd()
print(current_dir)
main_folder = current_dir +'/TrainedModels'

if not os.path.exists(main_folder):
    os.mkdir(main_folder)


# Create method folders
methods = ['/Euler', '/Euler/diffTau', '/Euler/Sym', '/Verlet', '/Verlet/diffTau', '/Verlet/Sym', '/Verlet/Sym/diffTau']

# All of dem problems
problems = ['/Kepler']

# General folders for each problem
folders = ['/ConvergenceGraphs', '/ConvergenceGraphs/Linear', '/ConvergenceGraphs/Sigmoid', '/MSE', '/Predictions', 
            '/Predictions/Linear', '/Predictions/Linear/T10', '/Predictions/Linear/T100', '/Predictions/Linear/T1000',
            '/Predictions/Sigmoid', '/Predictions/Sigmoid/T10', '/Predictions/Sigmoid/T100', '/Predictions/Sigmoid/T1000',
            '/PredictionsMultiple', '/PredictionsMultiple/Linear', '/PredictionsMultiple/Sigmoid', 
            '/ConvergenceGraphsMultiple', '/ConvergenceGraphsMultiple/Sigmoid', '/ConvergenceGraphsMultiple/Linear', '/SaveAtEpochs']

for method in methods:
    method = main_folder + method
    if not os.path.exists(method):
        print(method)
        os.mkdir(method)

    for problem in problems:
        problem = method + problem
        if not os.path.exists(problem):
            print(problem)
            os.mkdir(problem)

        for folder in folders:
            folder = problem + folder
            if not os.path.exists(folder):
                print(folder)
                os.mkdir(folder)


### Analysis folder
current_dir = os.getcwd()
if not os.path.exists(main_folder + '/Analysis'):
    os.mkdir(main_folder)

if not os.path.exists(main_folder + '/Analysis/Errors'):
    os.mkdir(main_folder)

if not os.path.exists(main_folder + '/Analysis/MSE_Acc'):
    os.mkdir(main_folder)

if not os.path.exists(main_folder + '/Analysis/VPT'):
    os.mkdir(main_folder)