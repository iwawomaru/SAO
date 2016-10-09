import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import Pong
from noh.components import GASuppressionBoosting


if __name__ == "__main__":

    n_stat = Pong.n_stat
    n_act = Pong.n_act
    n_learner = 4
    n_gene = 8
    eps_period = 5
    n_gen = 100
    one_gen = n_gene * eps_period
    n_eps = one_gen * n_gen

    model = GASuppressionBoosting.create(n_stat, n_act, n_learner, n_gene, eps_period)

    env = Pong(model, render=False)
    for i in xrange(n_eps):
        env.execute()
        if i%(one_gen*20) == 0:
            model.save('gasb_const_'+str(i/one_gen)+'.pkl')
