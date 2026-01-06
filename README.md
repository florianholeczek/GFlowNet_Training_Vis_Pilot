# Testing Visualizations for GFlowNet Training

The goal of this project is to develop visualizations to help developers of Generative Flow Networks in understanding and improving training.
Comparing logged samples to a testset representative of the state space helps assess the quality of the trained model and find areas to improve upon.
The dataset used for testing is created using the SEHTask, which rewards molecules with a high binding affinity to the sEH protein.


## Dashboard
The dashboard contains all visualizations with the possibility to crossfilter and displaying the image representation of the state on hover.

### Replay buffer
<img width="2528" height="1388" alt="Replay Buffer" src="https://github.com/user-attachments/assets/5d7e60e2-14af-4d1f-b8ed-4720e748d351" />
Shows the top n items based on reward for each iteration. This alows seeing when changes happen in the most interesting objects as well as a quick overview of the highest ranking objects.
Switching between reward and loss and highest/lowest will be added to allow answering the following questions:

- What are the highest ranking objects sampled?

- Does the range of the rewards match the testset reward range?

- What objects have the highest loss? These show where the model struggles and what parts of the state space are newly discovered

- What objects have the lowest loss? These areas of the state space have been visited more often.


### State space
<img width="2550" height="1360" alt="State space" src="https://github.com/user-attachments/assets/9dd414ba-741b-4cdc-a737-54c1d41d6fdf" />
Shows the final objects downprojected in two dimensions (In this case based on the fingerprints of the molecules). Adding the data of the testset is possible here.
This helps answer the questions:

- What parts of the state space covered by the testset are discovered by the model?

- What parts of the state space are discovered at which point in training?

### Trajectory explorer
<img width="2642" height="1450" alt="Trajectories" src="https://github.com/user-attachments/assets/8f8acba3-1e38-47db-99e2-e3bba7732c0a" />
Shows all states of all trajectories downprojected. Each line is one trajectory.
Might get removed, as downprojecting all states is very costly and grows exponentially with number of trajectories and trajectory length and no insights have been found yet.
Focus on the state space of the final objects might be better.

### DAG
<img width="3590" height="2420" alt="grafik" src="https://github.com/user-attachments/assets/4251f277-0ff1-413b-ba71-5194687d4bd8" />

This shows the Directed Acyclic Graph of the sampled objects. To reduce the size:

1. Linear chains (one parent one chains) have been truncated and their logprobabilities added up

2. Same Edges from different iterations have been merged into one

3. The DAG starts empty and can be expanded by selecting the objects from the table on the right

4. Alternatively final objects can be selected in other visualizations and their trajectories will be expanded (until deselection)

The overview on top shows the edges of the DAG with different metrics: highest/lowest logprobabilities, variance in logprobabilities or frequency. The edge coloring in the DAG fits the choosen metric.
This helps answering the following questions:

- How did the transition probabilities change during training?
  
- Which transitions increased/decreased most?
  
- How does the model sample trajectories in detail?
  
- How does the forward/backward policy differ in detail?



## How to run the dashboard
You will need [git lfs](https://git-lfs.com/) for the dataset and the images of the molecules.


Clone the repo and set up a venv using the requirements_db.txt file.
Python version 3.10 is tested.
Create a virtual environment:

Windows
```shell
py -m venv venv

# In cmd.exe
venv\Scripts\activate.bat
# Or In PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements_db.txt
```

Linux:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_db.txt
```

The repo contains only the first 1000 of 2500 images, so some objects might appear empty.
To solve this unpack the image.zip in the traindata1 folder into the folder images.
Image generation will be moved from files to on the fly generation.


Start the dashboard with:

```shell
# windows
python python\dashboard.py
# linux
python3 python/dashboard.py
```

## Logger
The logger.py allows logging during training in the format expected by the dashboard. 
It is currently outdated as I want to switch from logging the top reward objects during training to sampling n-objects on-policy after m iterations. 


## Notebook
I used the GFN_training.ipynb notebook to create the training data with this [library](https://github.com/recursionpharma/gflownet).
To use it follow its instructions to create the training venv.
