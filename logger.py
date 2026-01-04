import warnings

import pandas as pd
from typing import Callable
from datetime import datetime
import os
import torch
import numpy as np

#TODO: rewrite: Save as SQLite, add loss, from max reward to on policy stuff
#TODO: clean before writing to db: same consecutive texts should be merged. before image/feature calculation
"""
Use the provided logger to log n on-policy samples every m iterations.
The logger saves the data in a SQLite database to be queried by the dashboard.

Functions:
    log() stores all given data in a df.
    write_to_db() writes the logged data to the database after logging is done.
    
Scalability: 
    The complete trajectories get saved, so the resulting rowcount is n * (total iterations / m) * average trajectory length.
    Keeping this below 1e6 should work fine.
    Rule of thumb: 1000 samples at 20 timepoints during training should work fine for trajectories below length 50.
"""




class VisLogger:
    def __init__(
            self,
            path: str ="./",
            s0_included: bool = True,
            fn_state_to_text: Callable | None = None,
            fn_state_to_image: Callable | None = None,
            fn_compute_features: Callable | None = None,
            metrics: list[str] | None = None,
            features: list[str] | None = None
    ):
        """
        A Logger for the visualizations.
        :param path: path to a folder to save the data. If none creates based on datetime
        :param s0_included: if True, the trajectories are expected to have the start state included.
            The start states will then be removed before writing to the database.
             s0 will be specified as '#' in the visualizations
        :param fn_state_to_text: Optional.
            Function to convert a batch of states to a list of readable strings to identify a single state.
            Neccessary to distinguish states.
            Consecutive states with the same text will be merged (logprobs will be added).
            s0 will be specified as '#' in the visualizations, make sure no state has the same identifier.
        :param fn_state_to_image: Optional.
            Function to convert a batch of states to a list of images.
            The images are expected to be small pngs (<=200x200).
        :param fn_compute_features: Optional.
            Function to compute features from the states.
            Should take the states in the same format as given (torch Tensor, np array or list)
            and return a tuple consisting of:
                1. A np array of size (sum(trajectory_lengths), n_features) containing the features
                2. A list(bool) that tells for each state if the feature computation was successful.
                Unsuccessful states are skipped in the trajectory visualizations
            These will be used for the downprojections.
            Additional features can also be logged with the features parameter
        :param metrics: Optional.
            Names of additional metrics for final objects. Might be different losses or rewards.
            If the reward or loss function consists of multiple parts you can specify all of them here (list of strings).
            They will need to be logged each iteration.
            Otherwise only the total reward and the loss will be logged.
        :param features: Optional.
            If you want to log features, specify them here (list of strings).
            They will need to be logged each iteration.
            The features will be used for the downprojections.
            If features can be calculated from the states you can additionally use the fn_compute_features parameter.
        """

        if path:
            self.path = path
        else:
            self.path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.path)

        self.db = f"{self.path}/data.db"
        self.metrics = metrics
        self.features = features
        self. s0_included = s0_included
        self.fn_state_to_text = fn_state_to_text
        self.fn_state_to_image = fn_state_to_image
        self.fn_compute_features = fn_compute_features
        self.cols = [
            'final_id',
            'step',
            'final_object',
            'text',
            'iteration',
            'total_reward',
            *(metrics if metrics is not None else []),
            'loss'
            'logprobs_forward',
            'logprobs_backward',
            'image',
            'features_valid'
            *(features if features is not None else [])
        ]
        self.df = pd.DataFrame(columns=self.cols)

        # tracks the logged data
        self. current = {
            "batch_idx": None,
            "states": None,
            "rewards": None,
            "losses": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.current.update({name: None for name in metrics or []})
        self.current.update({name: None for name in features or []})

        # Warnings
        if not self.fn_state_to_text:
            warnings.warn("""
            States will not be distinguishable by text id. 
            DAG visualizations will not be meaningful. 
            Provide a state_to_text function when defining the logger to prevent that.
            """)
        if not self.fn_state_to_image:
            warnings.warn("""
            States will not be distinguishable by image. 
            Most visualizations will not allow for identifying a state . 
            Provide a state_to_image function when defining the logger to prevent that.
            """)

    def log(
            self,
            batch_idx: np.ndarray | torch.Tensor | None,
            states: np.ndarray | torch.Tensor | list | None,
            total_rewards: np.ndarray | torch.Tensor | None,
            losses: np.ndarray | torch.Tensor | None,
            iteration: int | None,
            logprobs_forward: np.ndarray | torch.Tensor | None,
            logprobs_backward: np.ndarray | torch.Tensor | None,
            metrics: np.ndarray | torch.Tensor | list[list[float]] | None = None,
            features: np.ndarray | torch.Tensor | list[list[float]] | None = None,
    ):
        """
        Logs everything provided for the current iteration.
        :param batch_idx: indices of final object of the batch.
            Each final object should have one index repeating for the length of its trajectory.
            [0,0,0,0,1,1,2,...] -> Object 0, trajectory length 4, Object 1, trajectory length 2 ...
            Trajectory data is expected to be logged from first state to last.
            In the example above the final objects would be at index 3 for the first Trajectory and
            index 5 for the second.
            Expected size: (sum(trajectory_lengths), ).
        :param total_rewards: array or tensor of total rewards.
            Expected size: (batchsize, ).
        :param losses: array or tensor of losses.
            Expected size: (batchsize, ).
        :param iteration: current iteration
        :param states: array or tensor of states (full trajectories).
            Expected size: (sum(trajectory_lengths), ).
        :param logprobs_forward: array or tensor of forward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
            The logprob of a state s is expected to be the logprob of reaching s, see example in lobprobs_backward.
        :param logprobs_backward: array or tensor of backward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
            The lobprob of a state s is expected to be the logprob of reaching s, eg.:
            state   | logprob_forward       | logprobs_backward
            s0      | 0                     | logprob(s1->s0)=0
            s1      | logprob(s0->s1)       | logprob(s2->s1)
            s2      | logprob(s1->s2)       | logprob(s3->s2)
            st      | logprob(s(t-1)->st)   | logprob(st->s(t-1))
        :param metrics: Optional. Additionally logged metrics of final objects based on the initialized metrics.
            Total reward and loss are logged seperately.
            A torch Tensor or np array of shape (len(metrics), batchsize,) or
            a nested list of the same shape.
        :param features: Optional. Additionally logged features based on the initialized features.
            A torch Tensor or np array of shape (len(features), sum(trajectory_lengths)) or
            a nested list of the same shape.
        :return: None
        """

        def to_np(x, dtype):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return x.astype(dtype, copy=False)

        if batch_idx is not None:
            self.current["batch_idx"] = to_np(batch_idx, np.int16)
        if states is not None:
            if isinstance(states, list):
                self.current["states"] = states
            elif isinstance(states, torch.Tensor):
                self.current["states"] = states.detach().cpu()
            else:
                self.current["states"] = to_np(states, np.float16)
        if total_rewards is not None:
            self.current["rewards"] = to_np(total_rewards, np.float16)
        if losses is not None:
            self.current["losses"] = to_np(losses, np.float16)
        if iteration is not None:
            self.current["iteration"] = int(iteration)
        if logprobs_forward is not None:
            self.current["logprobs_forward"] = to_np(logprobs_forward, np.float16)
        if logprobs_backward is not None:
            self.current["logprobs_backward"] = to_np(logprobs_backward, np.float16)
        if metrics is not None:
            for i,r in enumerate(self.rewards):
                self.current[r] = metrics[i]
        if features is not None:
            for i,r in enumerate(self.features):
                self.current[r] = features[i]

    def __check_data__(self):
        """
        Checks the data for completeness before writing to db.
        :return: None
        """
        for k, v in self.current.items():
            assert v is not None, f"{k} has not been logged"

        datalength = len(self.current["batch_idx"])
        for i in [self.current["states"], self.current["logprobs_forward"], self.current["logprobs_backward"]]:
            assert len(i) == datalength, f"lengths of batch_idx, logprobs and states must match"



    def write_to_db(self):
        """
        Writes the data of the current block to the database.
        Should be called every m iterations.
        :return: None
        """

        self.__check_data__()
        data = pd.DataFrame(columns=self.cols)

        # calculate the step for each state of a trajectory and if a state is final
        _, data["final_id"] = np.unique(self.current["batch_idx"], return_inverse=True)

        change = self.current["batch_idx"][1:] != self.current["batch_idx"][:-1]
        step = np.arange(len(self.current["batch_idx"])) - np.maximum.accumulate(
            np.r_[0, np.flatnonzero(change) + 1]
            .repeat(np.diff(np.r_[0, np.flatnonzero(change) + 1, len(self.current["batch_idx"])]))
        )
        last = np.r_[step[1:] == 0, True]
        data["step"] = step
        data["final_object"] = last

        # expand scalar iteration and add iteration and logprobs
        datalength = len(self.current["batch_idx"])
        data["iteration"] = np.array([self.current["iteration"]]*datalength)
        data["logprobs_forward"] = self.current["logprobs_forward"]
        data["logprobs_backward"] = self.current["logprobs_backward"]

        #expand arrays of length batchsize to db size (rewards, losses, provided additional metrics)
        nan_prefill = np.full(len(self.current["batch_idx"]), np.nan)
        metric_list = ["rewards", "losses"]
        if self.metrics is not None:
            metric_list += [i for i in self.metrics]
        for m in metric_list:
            ar = nan_prefill.copy()
            ar[last] = self.current[m]
            data[m] = ar

        # rename
        data.rename(columns={"rewards": "total_reward", "losses": "loss"}, inplace=True)

        # add provided features
        if self.features is not None:
            for i in self.features:
                data[i] = self.current[i]

        # delete s0 or update step to start with 1
        if self.s0_included:
            data = data[data["step"]!=0]
        else:
            data["step"] += 1

        # compute texts, images and features for states using given functions
        if self.fn_state_to_text is not None:
            data["text"] = self.fn_state_to_text(self.current["states"])
        else:
            data["text"] = "No text representation provided"

        if self.fn_state_to_image is not None:
            data["image"] = self.fn_state_to_image(self.current["states"])
        else:
            data["image"] = None
        #TODO image saving (Track number of rows of db?)

        if self.fn_compute_features is not None:
            features, features_valid = self.fn_compute_features(self.current["states"])
            data["features_valid"] = features_valid
            feature_cols = [f"computed_features_{i}" for i in range(features.shape[1])]
            features_df = pd.DataFrame(features, columns=feature_cols, index=data.index)
            data = pd.concat([data, features_df], axis=1)


        #TODO from here

        # TODO collapse same texts (handle no state to text fn)
        # TODO final ids shift and save to db (create first)

        # shift final ids for unique identifiers and write
        prev_data = pd.read_csv(self.csv)
        if not prev_data.empty:
            offset = prev_data["final_id"].max()+1
            data["final_id"] = data["final_id"] + offset
            data = pd.concat([prev_data, data])
        data.to_csv(self.csv, index=False)

        #TODO compute graphs and save nodes and edges db

        #reset
        self.current = {
            "batch_idx": None,
            "states": None,
            "rewards": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.current.update({name: None for name in self.rewards or []})
        self.current.update({name: None for name in self.features or []})



"""

def imagefn(inp):
    return ["image"]*len(inp)

def textfn(inp):
    return ["text"]*len(inp)

def featurefn(inp):
    return np.ones((len(inp),4))

logger = VisLogger(
    path="./",
    top_n=4,
    rewards=["r1", "r2", "r3"],
    features=["f1", "f2", "f3"],
    fn_state_to_image=imagefn,
    fn_compute_features=featurefn,
    fn_state_to_text=textfn,
)

for i in range(10):
    print("newround")

    logger.log(
        batch_idx=np.array([0,0,0,1,2,3,4,5,6,6,6,6,7]),
        total_rewards=np.arange(8)*(i+1),
        states=np.random.random(13),
        iteration=i,
        logprobs_forward=np.arange(13) * i,
        logprobs_backward=np.arange(13) * i,
        rewards = np.zeros((3,13)),
        features=np.zeros((3, 13))
    )
    print(logger.current)
    logger.finish_iteration()
    print(logger.top)
logger.write_to_db()

"""
