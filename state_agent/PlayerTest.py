from torch.distributions import Bernoulli

import torch
import torch
import pystk
from torch.distributions import Bernoulli
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from state_agent.player import network_features, action_to_actionspace, Team as Actor
from tournament.utils import VideoRecorder
from state_agent.Rollout_new import Rollout_new


if __name__ == "__main__":


    print("training done")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # inference
    team0 = Actor()
    #team0 = Jurgen()

    use_ray = False
    record_video = True
    video_name = "trained_dagger_agent.mp4"

    recorder = None
    if record_video:
        recorder = recorder & VideoRecorder(video_name)

    rollout = Rollout_new(team0=team0, use_ray=use_ray)

    rollout.__call__(use_ray=use_ray, record_fn=recorder)






