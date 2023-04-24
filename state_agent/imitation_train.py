import torch
from jurgen_agent.player import Team as Jurgen
from geoffrey_agent.player import Team as Geoffrey
from state_agent.player import network_features, Team as Actor
from state_agent.Rollout_new import rollout_many
from state_agent.planner import Planner
from state_agent.Rollout_new import Rollout_new
from os import path
from tournament.utils import VideoRecorder

def train():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_epochs = 10
    n_trajectories = 10
    batch_size = 128

    # Create the network
    action_net = Planner(13, 32, 3).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(action_net.parameters())

    # Create the losses
    mseLoss = torch.nn.MSELoss()
    bceLoss = torch.nn.BCEWithLogitsLoss()

    # Collect the data
    train_data = []
    for data in rollout_many([Jurgen()] * n_trajectories):
        train_data.extend(data)

    #print("size of train_data is ", len(train_data))
    #print("train_data[0] is ", train_data[0])
    #print("size of d[player_state] is ", len(train_data[0]['player_state']))

    features = network_features(train_data[0]['player_state'][0], train_data[0]['opponent_state'][0], train_data[0]['soccer_state'])
    print("original features shape is ", features.shape)
    print("ball velocity is ", train_data[0]['ball_velocity'])
    features_new = torch.cat([features, train_data[0]['ball_velocity']])
    print("new features shape is ", features_new.shape)

    train_features0 = torch.stack([torch.cat([network_features(d['player_state'][0], d['opponent_state'][0], d['soccer_state']), d['ball_velocity']]) for d
                                   in train_data]).to(device).float()
    print("train features (kart 0) shape: ", train_features0.shape)

    train_labels0 = torch.stack([torch.as_tensor((d['action0']['acceleration'], d['action0']['steer'], d['action0']['brake'])) for d in train_data]).to(device).float()
    print("train labels (kart 0) shape ", train_labels0.shape)

    train_features1 = torch.stack([torch.cat([network_features(d['player_state'][1], d['opponent_state'][1], d['soccer_state']), d['ball_velocity']]) for d
                                   in train_data]).to(device).float()
    print("train features (kart 1) shape: ", train_features1.shape)

    train_labels1 = torch.stack([torch.as_tensor((d['action1']['acceleration'], d['action1']['steer'], d['action1']['brake'])) for d in train_data]).to(device).float()
    print("train labels (kart 1) shape ", train_labels1.shape)

    train_features = torch.cat([train_features0, train_features1], dim=0)
    print("train features (both karts) shape: ", train_features.shape)

    train_labels = torch.cat([train_labels0, train_labels1], dim=0)
    print("train labels (both karts) shape ", train_labels.shape)

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
                print(batch_labels[:, :2].size())
                print(batch_labels[:, 2])

            o = action_net(batch_features)
            acc_loss_val = mseLoss(o[:, 0], batch_labels[:, 0])
            steer_loss_val = mseLoss(o[:, 1], batch_labels[:, 1])
            brake_loss_val = bceLoss(o[:, -1], batch_labels[:, -1])
            loss_val = 0.9*acc_loss_val + steer_loss_val + 0.1*brake_loss_val

            if iteration == 0 and epoch == 0:
                print(o.size())

            print("loss in iteration %d, epoch %d is %f" % (iteration/batch_size, epoch, loss_val))

            global_step += 1

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

    action_net.to("cpu")

    # Save the model
    model = torch.jit.script(action_net)
    torch.jit.save(model, 'my_traced_model.pt')

if __name__ == "__main__":

    # training
    train()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #inference
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

