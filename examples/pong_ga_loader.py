import sys, os, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import Pong
from noh.components import GASuppressionBoosting


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GA model loader')
    parser.add_argument('model', type=str, help='Model file path')
    args = parser.parse_args()

    n_stat = Pong.n_stat
    n_act = Pong.n_act
    n_learner = 4
    n_gene = 4
    eps_period = 5
    n_gen = 10
    one_gen = n_gene * eps_period
    n_eps = one_gen * n_gen

    model = GASuppressionBoosting.create(n_stat, n_act, n_learner, n_gene, eps_period)
    model.load(args.model)

    env = Pong(model, render=True)
    for i in xrange(n_eps):
        env.execute()
        if i%(one_gen*2) == 0:
            model.save('gasb_'+str(i/one_gen)+'gen.pkl')
    model.save('gasb_'+str(n_gen)+'gen.pkl')