
This repository contains the codebase of the paper titled "Solving Long-run Average Reward Robust MDPs via Stochastic Games".

1. Dependencies
In order to run the code the following dependencies must be met:

    - Python 3 should be installed. We used Python 3.9 for obtaining the results in the paper. 
    - `Numpy` library should be installed. 
    - `Stormpy` library should be installed. 
    - `matplotlib` library should be installed. 

2. Structure and How to run
There are four Python files in the repository.

    (i) `StrategyIteration.py` is the backend code, contatining the implementation of the RPPI algorithm described in the paper.
    
    (ii) `contamination.py` runs the experiments regarding the contamination model.
   
    (iii) `lake_unichain_priodic.py` runs the experiments regarding the unichain frozen lake model.
   
    (iv) `lake_multichain_priodic.py` runs the expeirments regarding the multichain frozen lake model.
   
The `results` folder contains the results we obtained by running the experiments (also in the paper). 

To run each of the experiments, simply execute: 
`python3 [experiment file]` 
where `[experiment file]` is one of (ii), (iii) or (iv) from the above list.  
