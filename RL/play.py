import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt
import os
import imageio

device = torch.device("cpu")

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

        self.model.load_state_dict(torch.load('./RL/model/police_net_490000_-81.pt',map_location=torch.device('cpu')))

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


class Labirint():
    def __init__(self, h, w, n_tree, n_state, coo1, coo2):
        self.h = h
        self.w = w
        self.n_tree = n_tree
        self.n_state = n_state
        self.findTree = []
        self.coo1 = coo1
        self.coo2 = coo2

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

        self.trees.append(self.coo1)
        self.trees.append(self.coo2)

        # while True:
        #     x = random.randint(0, self.h-1)
        #     y = random.randint(0, self.w-1)
        #
        #     if (x, y) in self.trees:
        #         continue
        #
        #     self.trees.append((x, y))
        #     if len(self.trees) == self.n_tree:
        #         break

        x = random.randint(0, self.h-1)
        y = random.randint(0, self.w-1)
        self.man = (x, y)
        return self.encode(self.man[0], self.man[1], self.trees[0][0], self.trees[0][1],
                           self.trees[1][0], self.trees[1][1])

    def updatevisual(self):
        # 1 - озеро, 2 - дерево, 3 - гриб, 8 - гном
        self.visual = np.zeros((self.h, self.w), dtype=int)

        for x, y in self.trees:
            if (x, y) in self.findTree:
                self.visual[x, y] = 6
            else:
                self.visual[x, y] = 3

        x, y = self.man
        self.visual[x, y] = 10

        return self.visual

    def render(self):
        self.updatevisual()
        print(np.array2string(self.visual, max_line_width=5*self.w).replace('0', '_'))
        # plt.imshow(self.visual, interpolation='none')

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



def reiforce(env, estimator_policy):
    states = []
    state = env.reset()
    n = 0
    filenames = []
    sumrew = 0
    while True:
        n += 1
        states.append(state)
        action, log_prob = estimator_policy.get_action(state)
        next_state, reward, is_done, act, param = env.step(action)
        sumrew += reward
        direction = ''
        if param == (0,1):
            direction = '→'
        if param == (0,-1):
            direction = '←'
        if param == (1,0):
            direction = '↓'
        if param == (-1,0):
            direction = '↑'
        if param == (1,1):
            direction = '→''↓'
        if param == (-1,-1):
            direction = '←''↑'
        if param == (-1,1):
            direction = '→''↑'
        if param == (1,-1):
            direction = '←''↓'

        env.updatevisual()

        filename = f'./RL/img/{n}.png'
        filenames.append(filename)
        ax1 = plt.subplot(121)
        ax1.imshow(env.visual, interpolation='none', cmap='hot')
        ax1.axis('off')

        ax2 = plt.subplot(122)
        text = f'Шаг: {n}\nДействие: {act}{direction}\nВознаграждение: {sumrew}\nОбработано: {len(env.findTree)} '
        ax2.text(0, 0.5, text)
        ax2.axis('off')

        plt.savefig(filename)
        plt.close()

        if is_done:
            with imageio.get_writer('mygif.gif', mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

            # Remove files
            for filename in set(filenames):
                os.remove(filename)

            break
        state = next_state


if __name__ == '__main__':
    h = 5
    w = 5
    n_tree = 2

    n_action = 9
    n_state = (h*w)**(1 + n_tree)
    n_hidden = 256
    # random.seed()
    env = Labirint(h, w, n_tree, n_state, (0,0), (1,3))
    # env.render()

    lr = 0.001
    police_net = PolicyNetwork(n_state, n_action, n_hidden, lr)

    reiforce(env, police_net)

    # torch.save(police_net.model.state_dict(), 'police_net_end.pt')
    # torch.save(value_net.model.state_dict(), 'value_net_end.pt')

