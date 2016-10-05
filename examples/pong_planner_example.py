import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh import Circuit
from noh.circuit import Planner, PropRule, TrainRule
from noh.components import Random, Const
from noh.environments import Pong

import numpy as np

n_stat = Pong.n_stat
n_act = Pong.n_act

component_set = []
component_set.append(Random(n_input=n_stat, n_output=n_act))
component_set.append(Const(n_input=n_stat, n_output=n_act, const_output=1))
component_set.append(Const(n_input=n_stat, n_output=n_act, const_output=2))
component_set.append(Const(n_input=n_stat, n_output=n_act, const_output=3))


class SimpleProp(PropRule):
    component_id_list = range(4)
    def __init__(self, components):
        super(SimpleProp, self).__init__(components)
        self.id = self.component_id_list.pop(0)

    def __call__(self, data):
        return self.components[self.id](data)

class EmplyProp(PropRule):
    def __init__(self, components):
        super(EmplyProp, self).__init__(components)
    def __call__(self, **kwargs):
        pass

class PFCPlanner(Planner):
    def __init__(self, components, rule_dict={}, default_prop=None, default_train=None):
        super(PFCPlanner, self).__init__(components, rule_dict, default_prop=None, default_train=None)
        self.f_go = False
        self.n_components = len(components)

    def __call__(self, data):
        if not self.f_go:
            self.prop_rule = np.random.choice(self.rules.values())
            self.f_go = True

        """ kashikoku shitai here """
        if np.random.rand() < 0.1:
            self.stop()

        return self.prop_rule(data)

    def train(self, data=None, label=None, epoch=None):
        pass

    def stop(self):
        self.f_go = False

    def supervised_train(self, data=None, label=None, epochs=None, **kwargs): pass
    def unsupervised_train(self, data=None, label=None, epochs=None, **kwargs): pass
    def reinforcement_train(self, data=None, label=None, epochs=None, **kwargs):  pass


if __name__ == "__main__":
    prop_rules = {}
    for i in xrange(4):
        prop_rules["prop"+str(i)] = SimpleProp


    model = Circuit(PFCPlanner, components=component_set, rule_dict=prop_rules,
                    default_prop=None, default_train=None)

    env = Pong(model, render=True)
    while True:
        env.execute()