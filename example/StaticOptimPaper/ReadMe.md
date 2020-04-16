To reproduce the figure of the paper Michaud (submitted) "Two efficient static optimization algorithms that account for muscle-tendon equilibrium: approaching the constraint Jacobian via a constant or a cubic spline function" follow these instructions:

Using an anaconda environment
1. Create a conda environment using the environment file (`conda env create -f environment.yml`)
1. Make sur your environment is activated (`conda activate static_optim`)
1. Install the static_optim package in the environment (`python setup.py install`)
1. Run the figure creation scripts (`python LinearApproximation.py` or `python ResultsComparison.py`)

Note the TimingArm26.txt file was created by hand. 
It is a collection of the timings took by each algorithm for 10 runs

