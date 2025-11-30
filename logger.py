import pandas as pd
from typing import Callable
import torch
import numpy as np


class VisLogger:
    def __init__(
            self,
            path: str ="./",
            top_n: int = 50,
            fn_state_to_text: Callable | None = None,
            fn_state_to_image: Callable | None = None,
            fn_compute_features: Callable | None = None,
            rewards: list[str] | None = None,
            features: list[str] | None = None
    ):
        """
        A Logger for the visualizations
        :param path: path to save the data
        :param top_n: the top n objects based on total reward will be written to the database.
        :param fn_state_to_text: Optional.
            Function to convert a batch of states to a list of readable strings
            to identify a single state.
            If neither this or fn_state_to_image are provided, visualized states can not be identified.
        :param fn_state_to_image: Optional.
            Function to convert a batch of states to a list of images.
            The images are expected to be base64 encoded strings in png or svg format.
            Their size can be small (<=200x200)
        :param fn_compute_features: Optional.
            Function to compute features from the states.
            Should take the states in the same format as given (torch Tensor, np array or list)
            and return a tuple consisting of:
                1. A np array of size (sum(trajectory_lengths), n_features) containing the features
                2. A list(bool) that tells for each state if the feature computation was successful.
                Unsuccessful states are skipped in the trajectory visualizations
            These will be used for the downprojections.
            Additional features can also be logged with the features parameter
        :param rewards: Optional.
            Names of intermediate rewards.
            If the reward function consists of multiple parts you can specify all of them here (list of strings).
            They will need to be logged each iteration.
            Otherwise only the total reward will be logged.
        :param features: Optional.
            If you want to log features, specify them here (list of strings).
            They will need to be logged each iteration.
            The features will be used for the downprojections.
            If features can be calculated from the states you can additionally use the fn_compute_features parameter.
        """
        self.csv = f"{path}logdata.csv"
        self.rewards = rewards
        self.features = features
        self.top_n = top_n
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
            *(rewards if rewards is not None else []),
            'logprobs_forward',
            'logprobs_backward',
            'image',
            *(features if features is not None else [])
        ]
        self.df = pd.DataFrame(columns=self.cols)
        self.df.to_csv(self.csv, index=False)

        # tracks the logged data
        self. current = {
            "batch_idx": None,
            "states": None,
            "rewards": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.current.update({name: None for name in rewards or []})
        self.current.update({name: None for name in features or []})
        # top items based on total reward
        self.top = {
            "batch_idx": None,
            "states": None,
            "rewards": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.top.update({name: None for name in rewards or []})
        self.top.update({name: None for name in features or []})

    def log(
            self,
            batch_idx: np.ndarray | torch.Tensor | None,
            states: np.ndarray | torch.Tensor | list | None,
            total_rewards: np.ndarray | torch.Tensor | None,
            iteration: int | None,
            logprobs_forward: np.ndarray | torch.Tensor | None,
            logprobs_backward: np.ndarray | torch.Tensor | None,
            rewards: np.ndarray | torch.Tensor | list[list[float]] | None = None,
            features: np.ndarray | torch.Tensor | list[list[float]] | None = None,
    ):
        """
        Logs everything provided for the current iteration.
        :param batch_idx: indices of final object of the batch.
            Each final object should have one index as repeating for the length of it's trajectory.
            [0,0,0,0,1,1,2,...] -> Object 0, trajectory length 4, Object 1, trajectory length 2
            Trajectory data is expected to be logged from first state to last.
            In the example above the final objects would be at index 3 for the first Trajectory and
            index 5 for the second.
            Expected size: (sum(trajectory_lengths), ).
        :param total_rewards: array or tensor of total rewards.
            Expected size: (batchsize, ).
        :param iteration: current iteration
        :param states: array or tensor of states (full trajectories).
            Expected size: (sum(trajectory_lengths), ).
        :param logprobs_forward: array or tensor of forward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
        :param logprobs_backward: array or tensor of backward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
        :param rewards: Optional. Additionally logged rewards based on the initialized rewards.
            Total reward is logged seperately.
            A torch Tensor or np array of shape (len(rewards), sum(trajectory_lengths)) or
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
        if iteration is not None:
            self.current["iteration"] = int(iteration)
        if logprobs_forward is not None:
            self.current["logprobs_forward"] = to_np(logprobs_forward, np.float16)
        if logprobs_backward is not None:
            self.current["logprobs_backward"] = to_np(logprobs_backward, np.float16)
        if rewards is not None:
            for i,r in enumerate(self.rewards):
                self.current[r] = rewards[i]
        if features is not None:
            for i,r in enumerate(self.features):
                self.current[r] = features[i]


    def finish_iteration(self):
        """
        Writes the data of the current iteration to self.top,
        keeping the top_n trajectories based on total reward.
        To be called at the end of each iteration after logging is done.
        :return: None
        """
        for k, v in self.current.items():
            assert v is not None, f"{k} has not been logged"

        datalength = len(self.current["batch_idx"])
        for i in [self.current["states"], self.current["logprobs_forward"], self.current["logprobs_backward"]]:
            assert len(i) == datalength, f"lengths of batch_idx, logprobs and states must match"

        if self.top["rewards"] is not None: #not first
            reward_cutoff = self.top["rewards"].min()
            mask = self.current["rewards"] > reward_cutoff
            filtered = self.current["rewards"][mask]
            if len(filtered) >0: #otherwise no new objects to add to top

                # combine rewards
                combined = np.concatenate((self.top["rewards"], filtered), axis=0)
                idx = np.argpartition(combined, -self.top_n)[-self.top_n:]
                self.top["rewards"] = combined[idx]


                mask_t = mask[self.current["batch_idx"]]
                filtered = self.current["batch_idx"][mask_t]

                # add value to have distinct values from top
                filtered += len(self.top["batch_idx"])
                combined = np.concatenate((self.top["batch_idx"], filtered), axis=0)
                _, combined = np.unique(combined, return_inverse=True)
                idx = np.isin(combined, idx)
                self.top["batch_idx"] = combined[idx]
                self.current["iteration"] = np.array([self.current["iteration"]]*datalength)

                if isinstance(self.current["states"], list):
                    masked = [x for x, keep in zip(self.current["states"], mask_t) if keep]
                    combined = self.top["states"] + masked
                    self.top["states"] = [x for x, keep in zip(combined, idx) if keep]
                else:
                    self.top["states"] = np.concatenate((self.top["states"], self.current["states"][mask_t]), axis=0)[idx]

                for i in [
                    "logprobs_forward",
                    "logprobs_backward",
                    "iteration",
                    *(self.features or []),
                    *(self.rewards or [])
                ]:
                    self.top[i] = np.concatenate((self.top[i], self.current[i][mask_t]), axis=0)[idx]

        else: # create first time
            idx = np.argpartition(self.current["rewards"], -self.top_n)[-self.top_n:]
            self.top["rewards"] = self.current["rewards"][idx]

            trajectory_idx = np.isin(self.current["batch_idx"], idx)
            self.top["iteration"] = np.array([self.current["iteration"]]*datalength)[trajectory_idx]

            if isinstance(self.current["states"], list):
                self.top["states"] = [x for x, keep in zip(self.current["states"], trajectory_idx) if keep]
            else:
                self.top["states"] = self.current["states"][trajectory_idx]

            for i in  [
                "logprobs_forward",
                "logprobs_backward",
                "batch_idx",
                *(self.features or []),
                *(self.rewards or [])
            ]:
                self.top[i] = self.current[i][trajectory_idx]


    def write_to_db(self):
        """
        Writes the data of the current block to the database.
        Should be called every n iterations. (rule of thumb n=500)
        :return: None
        """
        data = pd.DataFrame(columns=self.cols)
        _, data["final_id"] = np.unique(self.top["batch_idx"], return_inverse=True)

        change = self.top["batch_idx"][1:] != self.top["batch_idx"][:-1]
        step = np.arange(len(self.top["batch_idx"])) - np.maximum.accumulate(
            np.r_[0, np.flatnonzero(change) + 1]
            .repeat(np.diff(np.r_[0, np.flatnonzero(change) + 1, len(self.top["batch_idx"])]))
        )
        last = np.r_[step[1:] == 0, True]
        data["step"] = step
        data["final_object"] = last

        if self.fn_state_to_text is not None:
            data["text"] = self.fn_state_to_text(self.top["states"])
        else:
            data["text"] = "No text representation provided"

        nan_prefill = np.full(len(self.top["batch_idx"]), np.nan)
        rewards = nan_prefill.copy()
        rewards[last] = self.top["rewards"]
        data["total_reward"] = rewards

        data["iteration"] = self.top["iteration"]
        data["logprobs_forward"] = self.top["logprobs_forward"]
        data["logprobs_backward"] = self.top["logprobs_backward"]

        if self.fn_state_to_image is not None:
            data["image"] = self.fn_state_to_image(self.top["states"])

        if self.fn_compute_features is not None:
            features, features_valid = self.fn_compute_features(self.top["states"])
            data["features_valid"] = features_valid
            feature_cols = [f"computed_features_{i}" for i in range(features.shape[1])]
            features_df = pd.DataFrame(features, columns=feature_cols, index=data.index)
            data = pd.concat([data, features_df], axis=1)

        if self.rewards is not None:
            for i in self.rewards:
                data[i] = self.top[i]
        if self.features is not None:
            for i in self.features:
                data[i] = self.top[i]

        # shift final ids for unique identifiers and write
        prev_data = pd.read_csv(self.csv)
        if not prev_data.empty:
            offset = prev_data["final_id"].max()+1
            data["final_id"] = data["final_id"] + offset
            data = pd.concat([prev_data, data])
        data.to_csv(self.csv, index=False)

        #reset
        self.current = {
            "batch_idx": None,
            "states": None,
            "rewards": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.top = {
            "batch_idx": None,
            "states": None,
            "rewards": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.current.update({name: None for name in self.rewards or []})
        self.current.update({name: None for name in self.features or []})
        self.top.update({name: None for name in self.rewards or []})
        self.top.update({name: None for name in self.features or []})



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
