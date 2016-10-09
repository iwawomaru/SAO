from noh import Component
import random

'''
@author Ashihara Alex Aressuyo

This is not Neural Network
This Component only do Q-function learning

'''

class Qlearn(Component):
    '''
    n_input : Number of state from environment
    n_output : Number of actions ( f.g. Pong is 6)
    epsilon : e-greedy method to use (default 0.1)
    alpha : learning rate
    gamma : mou mendokuse
    '''

    #initialize
    def __init__(self, n_input, n_output, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}
        self.n_output = n_output
        self.n_input = n_input
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        print(self.n_output)
        print(self.n_input)

    # get Q-function
    def getQ(self, n_input, n_output):
        return self.q.get((n_input, n_output), 0.0)

    # update Q-funciton
    def learnQ(self, state, action, reward, value):
        old_Q = self.q.get((state, action), None)
        if old_Q is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = old_Q + self.alpha * (value - old_Q)

    # choose action followed by Q-function
    def chooseAction(self, return_q=False):
        q = [self.getQ(self.n_input, a) for a in xrange(self.n_output)]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random()* mag - .5 * mag for i in xrange(self.n_output)]
            maxQ = max(q)

        count = q.count(maxQ)

        # in case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in xrange(self.n_output) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = i 
        if return_q:
            return action, q
        
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in xrange(self.n_output)])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    
        
    def __call__(self, data, **kwargs):
        return self.chooseAction()
        
    def supervised_train(self, data=None, label=None, epochs=None, **kwargs): pass
    def unsupervised_train(self, data=None, label=None, epochs=None, **kwargs): pass

    def reinforcement_train(self, state, reward, data=None, label=None, epochs=None, **kwargs):
        selectedAction = return_action()
        self.learn(self.n_input, self.n_output, reward, state)
