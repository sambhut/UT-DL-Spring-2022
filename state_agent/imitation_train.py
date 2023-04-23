import torch
from jurgen_agent.player import Team as Jurgen
from state_agent.player import network_features, Team as Actor
from state_agent.Rollout_new import rollout_many
from state_agent.planner import Planner
from state_agent.Rollout_new import Rollout_new
from os import path
from tournament.utils import VideoRecorder

def train():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_epochs = 5
    n_trajectories = 10
    batch_size = 128

    # Create the network
    action_net = Planner(11, 32, 3).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(action_net.parameters())

    # Create the loss
    loss = torch.nn.MSELoss()

    # Collect the data
    train_data = []
    for data in rollout_many([Jurgen()] * n_trajectories):
        train_data.extend(data)

    #print("size of train_data is ", len(train_data))
    #print("train_data[0] is ", train_data[0])
    #print("size of d[player_state] is ", len(train_data[0]['player_state']))

    train_features = torch.stack([network_features(d['player_state'][0], d['opponent_state'][0], d['soccer_state']) for d in train_data]).to(device).float()

    train_labels = torch.stack([torch.as_tensor((d['action']['acceleration'], d['action']['steer'], d['action']['brake'])) for d in train_data]).to(device).float()

    # Start training
    global_step = 0
    action_net.train().to(device)

    for epoch in range(n_epochs):
        for iteration in range(0, len(train_data), batch_size):
            batch_ids = torch.randint(0, len(train_data), (batch_size,), device=device)
            batch_features = train_features[batch_ids]
            batch_labels = train_labels[batch_ids]

            if iteration == 0 and epoch == 0:
                print(batch_features)
                print(batch_features.size())
                print(batch_labels)
                print(batch_labels.size())

            o = action_net(batch_features)
            loss_val = loss(o, batch_labels)

            print("loss in iteration %d, epoch %d is %f" % (iteration/batch_size, epoch, loss_val))

            global_step += 1

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

    action_net.to("cpu")

    # Save the model
    example_input = torch.randn(1, 11)

    traced_model = torch.jit.trace(action_net, example_input)
    torch.jit.save(traced_model, 'my_traced_model.pt')

if __name__ == "__main__":

    train()

    print("training done")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    team0 = Actor()
    #team0 = Jurgen()

    use_ray = False
    record_video = True
    video_name = "trained_imitation_agent.mp4"

    recorder = None
    if record_video:
        recorder = recorder & VideoRecorder(video_name)

    rollout = Rollout_new(team0=team0, use_ray=use_ray)

    rollout.__call__(use_ray=use_ray, record_fn=recorder)

