import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math

device = torch.device("cuda:1")

class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            # nn.Linear(n_hidden, 64),
            # nn.ReLU(),
            nn.Linear(n_hidden, n_action),
            nn.Softmax(dim=1),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.n_state = n_state

    def predict(self, s):
        return self.model(s.to(device).to(torch.float32))[0]

    def update(self, returns, log_probs):
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(-log_prob * Gt)
        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        probs = self.predict(F.one_hot(torch.tensor([s]).to(torch.long), self.n_state))
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob


class ValueNetwork():
    def __init__(self, n_state, n_hidden, lr=0.05):
        self.criterion = nn.MSELoss()
        self.model = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            # nn.Linear(n_hidden, 64),
            # nn.ReLU(),
            nn.Linear(n_hidden, 1)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.n_state = n_state

    def update(self, s, y):
        y_pred = self.model(F.one_hot(torch.tensor([s]).to(torch.long), self.n_state).to(torch.float32).to(device)).squeeze(1)
        loss = self.criterion(y_pred.squeeze(), Variable(torch.Tensor(y).to(device)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(F.one_hot(torch.tensor([s]), self.n_state).squeeze(0).to(torch.float32).to(device))

class Labirint():
    def __init__(self, h, w, n_tree, n_state):
        self.h = h
        self.w = w
        self.n_tree = n_tree
        self.n_state = n_state
        self.findTree = []

        self.reset()

    def encode(self, man_row, man_col, tree_row1, tree_col1, tree_row2, tree_col2):
        # (10) 10, 10, 10, 2
        i = man_row
        i *= self.h
        i += man_col
        i *= self.w
        i += tree_row1
        i *= self.h
        i += tree_col1
        i *= self.w
        i += tree_row2
        i *= self.h
        i += tree_col2
        return i


    def reset(self):
        self.findTree = []
        self.trees = []

        while True:
            x = random.randint(0, self.h-1)
            y = random.randint(0, self.w-1)

            if (x, y) in self.trees:
                continue

            self.trees.append((x, y))
            if len(self.trees) == self.n_tree:
                break

        self.mushrooms = []
        x = random.randint(0, self.h-1)
        y = random.randint(0, self.w-1)
        self.man = (x, y)
        return self.encode(self.man[0], self.man[1], self.trees[0][0], self.trees[0][1],
                           self.trees[1][0], self.trees[1][1])

    def updatevisual(self):
        self.visual = np.zeros((self.h, self.w), dtype=int)

        for x, y in self.trees:
            self.visual[x, y] = 2
        x, y = self.man
        self.visual[x, y] = 8

        return self.visual

    def render(self):
        self.updatevisual()
        print(np.array2string(self.visual, max_line_width=5*self.w).replace('0', '_'))

    def step(self, action):
        '''new_state
        reward
        is_done'''
        param = (-1, -1)
        if action == 0:
            act = 'fix'
        else:
            act = 'go'
            if action == 1:
                param = (1, 0)
            if action == 2:
                param = (0, 1)
            if action == 3:
                param = (-1, 0)
            if action == 4:
                param = (0, -1)
            if action == 5:
                param = (1, 1)
            if action == 6:
                param = (-1, -1)
            if action == 7:
                param = (-1, 1)
            if action == 8:
                param = (1, -1)

            self.man = tuple(a+b for a, b in zip(param, self.man))
            self.man = (min(max(0, self.man[0]), self.h - 1), min(max(0, self.man[1]), self.w - 1))

        is_done = False
        if act == 'go':
            reward = -1
        elif act == 'fix':
            if self.man in self.trees:
                if self.man not in self.findTree:
                    reward = 20
                    self.findTree.append(self.man)
                else:
                    reward = -10

            else:
                reward = -10
        if len(self.findTree) == self.n_tree:
            is_done = True

        new_state = self.encode(self.man[0], self.man[1], self.trees[0][0], self.trees[0][1],
                                self.trees[1][0], self.trees[1][1])

        return new_state, reward, is_done, act, param



def reiforce(env, estimator_policy, estimator_value, n_episode, gamma=1):
    for episode in range(n_episode):
        states = []
        log_probs = []
        rewards = []
        state = env.reset()
        n = 0
        while True:
            n += 1
            states.append(state)
            action, log_prob = estimator_policy.get_action(state)
            next_state, reward, is_done, act, param = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)
            # print(act, param, reward)
            if is_done or n == 1500:
                returns = []
                Gt = 0
                pw = 0
                for t in range(len(states)-1, -1, -1):
                    Gt += gamma ** pw * rewards[t]
                    pw += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.tensor(returns)
                baseline_values = estimator_value.predict(states)
                advantages = returns.to(device) - baseline_values
                estimator_value.update(states, returns)
                estimator_policy.update(advantages, log_probs)
                if episode == 0:
                    mean_reward = 0
                else:
                    mean_reward = sum(total_reward_episode[max(0, episode-1000): episode])/len(total_reward_episode[max(0, episode-1000): episode])
                # mean_reward = sum(total_reward_episode[max(0, episode-1000): episode])/  len(total_reward_episode[max(0, episode-1000): episode)
                print('Эпизод: {}, полное вознаграждение: {}, число шагов {}, среднее {:.1f}'.format(episode,
                                        total_reward_episode[episode], len(states), mean_reward
                                    ))
                if episode != 0 and episode % 10_000 == 0:
                    torch.save(police_net.model.state_dict(), './model/police_net_{}_{:.0f}.pt'.format(episode,
                                                    mean_reward))
                    torch.save(value_net.model.state_dict(), './model/value_net_{}_{:.0f}.pt'.format(episode,
                                                    mean_reward))

                # n += 1
                break
            state = next_state
            # print(n, act, param)
            # env.render()
            # pass


if __name__ == '__main__':
    h = 5
    w = 5
    n_tree = 2

    n_action = 9
    n_state = (h*w)**(1 + n_tree) #
    n_hidden = 256
    env = Labirint(h, w, n_tree, n_state)
    # env.render()

    # n_action - fix, go на любую клетку поля

    n_episode = 500_000
    lr = 0.0005
    police_net = PolicyNetwork(n_state, n_action, n_hidden, lr)

    n_hidden_v = 256
    lr_v = 0.0005
    value_net = ValueNetwork(n_state, n_hidden_v, lr_v)
    gamma = 0.99
    total_reward_episode = [0] * n_episode
    reiforce(env, police_net, value_net, n_episode, gamma)

    torch.save(police_net.model.state_dict(), 'police_net_end.pt')
    torch.save(value_net.model.state_dict(), 'value_net_end.pt')

    plt.plot(total_reward_episode)
    plt.show()

