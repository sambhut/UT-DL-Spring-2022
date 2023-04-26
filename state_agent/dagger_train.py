import torch
from jurgen_agent.player import Team as Jurgen
from geoffrey_agent.player import Team as Geoffrey
from state_agent.player import network_features, Team as Actor
from state_agent.Rollout_new import rollout_many
from state_agent.planner import Planner
from state_agent.Rollout_new import Rollout_new
from os import path
from tournament.utils import VideoRecorder
import time

print_val = 0

def train():
    global print_val

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_epochs = 20
    n_trajectories = 10
    batch_size = 128
    learning_rate1 = 0.001
    learning_rate2 = 0.000001
    #weight_decay = 1e-5
    weight_decay1 = 0
    weight_decay2 = 0

    expert_agent = Jurgen()
    #expert_agent = Jurgen()

    # Create the network
    action_net = Planner(13, 128, 3).to(device)

    # Create the optimizer
    optimizer1 = torch.optim.Adam(action_net.parameters(), lr=learning_rate1, weight_decay=weight_decay1)
    #optimizer1 = torch.optim.SGD(action_net.parameters(), lr=learning_rate1, momentum=0.9, weight_decay=weight_decay1)

    optimizer2 = torch.optim.Adam(action_net.parameters(), lr=learning_rate2, weight_decay=weight_decay2)
    #optimizer2 = torch.optim.SGD(action_net.parameters(), lr=learning_rate2, momentum=0.9, weight_decay=weight_decay2)

    # Create the losses
    mseLoss = torch.nn.MSELoss()
    bceLoss = torch.nn.BCEWithLogitsLoss()

    # Collect the data for imitation learning
    train_data_imitation = []
    for data in rollout_many([expert_agent] * n_trajectories):
        train_data_imitation.extend(data)

    imitation_features = network_features(train_data_imitation[0]['player_state'][0], train_data_imitation[0]['opponent_state'][0],
                                train_data_imitation[0]['soccer_state'])
    print("original imitation features shape is ", imitation_features.shape)
    print("ball velocity is ", train_data_imitation[0]['ball_velocity'])
    features_new = torch.cat([imitation_features, train_data_imitation[0]['ball_velocity']])
    print("new imitation features shape is ", features_new.shape)

    train_features_imitation0 = torch.stack([torch.cat([network_features(d['player_state'][0], d['opponent_state'][0], d['soccer_state']), d['ball_velocity']]) for d
                                   in train_data_imitation]).to(device).float()
    print("train features imitation (kart 0) shape: ", train_features_imitation0.shape)

    train_labels_imitation0 = torch.stack([torch.as_tensor((d['action0']['acceleration'], d['action0']['steer'], d['action0']['brake'])) for d in train_data_imitation]).to(device).float()
    print("train labels imitation (kart 0) shape ", train_labels_imitation0.shape)

    train_features_imitation1 = torch.stack([torch.cat([network_features(d['player_state'][1], d['opponent_state'][1], d['soccer_state']), d['ball_velocity']]) for d
                                   in train_data_imitation]).to(device).float()
    print("train features imitation (kart 1) shape: ", train_features_imitation1.shape)

    train_labels_imitation1 = torch.stack([torch.as_tensor((d['action1']['acceleration'], d['action1']['steer'], d['action1']['brake'])) for d in train_data_imitation]).to(device).float()
    print("train labels imitation (kart 1) shape ", train_labels_imitation1.shape)

    train_features_imitation = torch.cat([train_features_imitation0, train_features_imitation1], dim=0)
    print("train features imitation (both karts) shape: ", train_features_imitation.shape)

    train_labels_imitation = torch.cat([train_labels_imitation0, train_labels_imitation1], dim=0)
    print("train labels imitation (both karts) shape ", train_labels_imitation.shape)

    # Start training based on imitation learning
    global_step = 0
    action_net.train().to(device)

    for epoch in range(n_epochs):
        for iteration in range(0, len(train_data_imitation), batch_size):
            batch_ids = torch.randint(0, len(train_data_imitation), (batch_size,), device=device)
            batch_features = train_features_imitation[batch_ids]
            batch_labels = train_labels_imitation[batch_ids]

            o_acc, o_steer, o_brake = action_net(batch_features)
            acc_loss_val = mseLoss(o_acc[:, 0], batch_labels[:, 0])
            steer_loss_val = mseLoss(o_steer[:, 0], batch_labels[:, 1])
            brake_loss_val = bceLoss(o_brake[:, 0], batch_labels[:, -1])
            loss_val = 0.9*acc_loss_val + steer_loss_val + 0.1*brake_loss_val

            print("imitation loss in iteration %d, epoch %d is %f" % (iteration/batch_size, epoch, loss_val))

            global_step += 1

            optimizer1.zero_grad()
            loss_val.backward()
            optimizer1.step()

    action_net.to("cpu")

    # Save the model as act expects the pt file
    model = torch.jit.script(action_net)
    torch.jit.save(model, 'my_traced_model.pt')


# Collect the data for dagger
    train_data_dagger = []
    for data in rollout_many([Actor()] * n_trajectories):
        train_data_dagger.extend(data)

    train_features_dagger0 = torch.stack([torch.cat([network_features(d['player_state'][0], d['opponent_state'][0], d['soccer_state']), d['ball_velocity']]) for d
                                   in train_data_dagger]).to(device).float()
    print("train features dagger (kart 0) shape: ", train_features_dagger0.shape)

    print_val = train_data_dagger[0]['action0']

    train_features_dagger1 = torch.stack([torch.cat([network_features(d['player_state'][1], d['opponent_state'][1], d['soccer_state']), d['ball_velocity']]) for d
                                   in train_data_dagger]).to(device).float()
    print("train features dagger (kart 1) shape: ", train_features_dagger1.shape)

    train_features_dagger = torch.cat([train_features_dagger0, train_features_dagger1], dim=0)
    print("train features dagger (both karts) shape: ", train_features_dagger.shape)

    expert_agent.new_match(0, 2)

    action_dict = []
    for d in train_data_dagger:
        action_dict.append(expert_agent.act(d['player_state'], d['opponent_state'], d['soccer_state']))

    print(action_dict[0])

    train_labels_dagger0 = torch.stack([torch.as_tensor((l[0]['acceleration'], l[0]['steer'], l[0]['brake'])) for l in action_dict]).to(device).float()
    print("train labels dagger (kart 0) shape: ", train_labels_dagger0.shape)

    train_labels_dagger1 = torch.stack([torch.as_tensor((l[1]['acceleration'], l[1]['steer'], l[1]['brake'])) for l in action_dict]).to(device).float()
    print("train labels dagger (kart 1) shape: ", train_labels_dagger1.shape)

    train_labels_dagger = torch.cat([train_labels_dagger0, train_labels_dagger1], dim=0)
    print("train labels dagger (both karts) shape ", train_labels_dagger.shape)

    # merge imitation learning and dagger data
    total_train_features = torch.cat([train_features_imitation, train_features_dagger], dim=0)
    total_train_labels = torch.cat([train_labels_imitation, train_labels_dagger], dim=0)

    print("size of total training features is ", total_train_features.size())
    print("size of total training labels is ", total_train_labels.size())

    # Start training for dagger using combined data
    global_step = 0
    action_net.train().to(device)

    for epoch in range(n_epochs):
        for iteration in range(0, len(total_train_features), batch_size):
            batch_ids = torch.randint(0, len(total_train_features), (batch_size,), device=device)
            batch_features = total_train_features[batch_ids]
            batch_labels = total_train_labels[batch_ids]

            o_acc, o_steer, o_brake = action_net(batch_features)
            acc_loss_val = mseLoss(o_acc[:, 0], batch_labels[:, 0])
            steer_loss_val = mseLoss(o_steer[:, 0], batch_labels[:, 1])
            brake_loss_val = bceLoss(o_brake[:, 0], batch_labels[:, -1])

            # Assign different weights for each loss
            loss_val = 0.9*acc_loss_val + steer_loss_val + 0.1*brake_loss_val

            print("dagger training loss in iteration %d, epoch %d is %f" % (iteration/batch_size, epoch, loss_val))

            global_step += 1

            optimizer2.zero_grad()
            loss_val.backward()
            optimizer2.step()

    action_net.to("cpu")

    # Save the final model
    model = torch.jit.script(action_net)
    torch.jit.save(model, 'my_traced_model.pt')

if __name__ == "__main__":

    start_time = time.time()

    # training
    train()

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

    print(print_val)

    print(" Total execution time = %.2f seconds" %(time.time()-start_time))