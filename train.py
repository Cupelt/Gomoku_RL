import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model.agent as model
import model.environment as env

path = './trined/model_state_dict.pt'
save_rate = 5

table_size = (15, 15) # only 15*15 supported

device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.001

eps_start = 1.0
eps_end = 0.01
eps_decay = 0.99975

gamma = .99
tau = 0.005

agent_A = model.Agent(table_size).to(device)
agent_B = model.Agent(table_size).to(device)

optim_A = optim.Adam(agent_A.parameters(), learning_rate)
optim_B = optim.Adam(agent_B.parameters(), learning_rate)

criterion = nn.SmoothL1Loss()

if os.path.exists(path):
    i_episode, state_dict_A, state_dict_B = model.model_loader(path)

    agent_A.load_state_dict(state_dict_A)
    agent_B.load_state_dict(state_dict_B)

else:
    i_episode = 0

def select_action(model, state, episode):
    eps_threshold = max(eps_end, eps_start * (eps_decay ** episode))
    noise = torch.rand(table_size[0] * table_size[1], device=device) * eps_threshold
    
    with torch.no_grad():
        pred = model(state)
        return pred, pred.clone(), torch.argmax(pred * noise).view(-1, 1)


while True:

    # reset env
    state = torch.zeros(table_size, device=device).view(1, 1, table_size[0], table_size[1])

    # shuffle agents
    agents = [(agent_A, optim_A), (agent_B, optim_B)]
    random.shuffle(agents)

    done = False

    #start episode
    while not done:

        # competition between models
        for AoD, _agent in enumerate(agents):
            
            _agent, optimizer = _agent
            pred_Q, target_Q, action = select_action(_agent, state, i_episode)

            next_state, reward, done = env.step(table_size, state.cpu().view(15, 15).numpy(), AoD, action.item())
            reward = torch.tensor(reward, dtype=torch.float32).to(device)

            if done:
                target_Q[action] = reward
            else:
                new_Qpred = _agent(next_state).clone().detach()
                target_Q[action] = reward + gamma * torch.max(new_Qpred)

            loss = criterion(pred_Q, target_Q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state


    print("Episode {:4d}, Winner ".format(i, np.sum(local_loss)/len(local_loss)))

