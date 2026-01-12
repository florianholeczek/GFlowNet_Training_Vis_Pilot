# Testing Visualizations for GFlowNet Training

The goal of this project is to develop visualizations to help developers of Generative Flow Networks in understanding and improving training.
Comparing logged samples to a testset representative of the state space helps assess the quality of the trained model and find areas to improve upon.
The dataset used for testing is created using the SEHTask, which rewards molecules with a high binding affinity to the sEH protein.


## Dashboard
The dashboard contains all visualizations with the possibility to crossfilter and displaying the image representation of the state on hover.

### Sampled Objects Overview
<img width="2550" height="968" alt="grafik" src="https://github.com/user-attachments/assets/86bd4e01-6266-4f91-ae1e-adf5e74ad7ff" />

Shows the top n items based on reward, loss or other provided metrics for each iteration. This allows seeing when changes happen in the most interesting objects as well as a quick overview of the highest ranking objects.
Views are either persistent (highest/lowest ranking cumulative over iterations) or highest/lowest per iteration.
This allows the following questions:

- What are the highest ranking objects sampled?

- When did the model discover high reward areas? The plot allows to distinguish between phases in traning where much / little change happened

- What objects have the highest loss? These show where the model struggles and what parts of the state space are newly discovered

- What objects have the lowest loss? These areas of the state space have been visited more often.


### State space
<img width="2550" height="1360" alt="State space" src="https://github.com/user-attachments/assets/9dd414ba-741b-4cdc-a737-54c1d41d6fdf" />
Shows the final objects downprojected in two dimensions (In this case based on the fingerprints of the molecules). Adding the data of the testset is possible here.
This helps answer the questions:

- What parts of the state space covered by the testset are discovered by the model? Did the model learn a sufficient part of the state space?

- What parts of the state space are discovered at which point in training?

- In what areas does the model struggle to sample proportonally (areas with highest loss)

### DAG
<img width="3590" height="2420" alt="grafik" src="https://github.com/user-attachments/assets/4251f277-0ff1-413b-ba71-5194687d4bd8" />

This shows the Directed Acyclic Graph of the sampled objects. To reduce the size:

1. Linear chains (one parent one chains) have been truncated and their logprobabilities added up

2. Same Edges from different iterations have been merged into one

3. The DAG starts empty and can be expanded by selecting the objects from the table on the right

4. Alternatively final objects can be selected in other visualizations and their trajectories will be expanded (until deselection) Selecting a node in the DAG expands its trajectories as well.

The overview on top shows the edges of the DAG with different metrics: highest/lowest logprobabilities, variance in logprobabilities or frequency. The edge coloring in the DAG fits the choosen metric.
This helps answering the following questions:

- How did the transition probabilities change during training?
  
- Which transitions increased/decreased most?
  
- How does the model sample trajectories in detail?
  
- How does the forward/backward policy differ in detail?



## How to run the dashboard with the seh data
You will need [git lfs](https://git-lfs.com/) for the dataset. Alternatively just download the zip of the seh_small folder


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

Start the dashboard with:

```shell
# windows
python python\main.py
# linux
python3 python/main.py
```

## Logger
The logger.py allows logging during training in the format expected by the dashboard. 
Find the documentation directly in the file.

# Running the dashboard on your own logged data

Log your training with the logger and add the testset if neccessary.
You will need an text to image function that converts your logged text representations of the state to images to identify a state. Run the db like this:

```python
from dashboard import run_dashboard
run_dashboard(data="FolderOfYourLoggedData", text_to_img_fn=your_text_to_img_fn)
```


## Notebook
I used the GFN_training.ipynb notebook to create the training data with this [library](https://github.com/recursionpharma/gflownet).
To use it follow its instructions to create the training venv.
