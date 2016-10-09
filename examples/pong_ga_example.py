import sys, os, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import Pong
from noh.components import GASuppressionBoosting


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GA model trainer')
    parser.add_argument('prefix', '-p', default='gasb_const', type=str, help='Model prefix')
    parser.add_argument('mu_scale', '-m', default=0.6, type=float, help='Scale of mu')
    parser.add_argument('sigma', '-s', default=0.001, type=float, help='Sigma')
    parser.add_argument('genes', '-g', default=8, type=int, help='# of genes')
    parser.add_argument('eps_period', '-e', default=5, type=int, help='episode period of genes')
    parser.add_argument('--dqn', action='store_true', default=False, help='DQN mode flag')
    args = parser.parse_args()

    n_stat = Pong.n_stat
    n_act = Pong.n_act
    n_learner = 4
    n_gen = 100
    one_gen = n_gene * args.eps_period
    n_eps = one_gen * n_gen
    isDqn = 'dqn' if args.dqn else 'random'

    model = GASuppressionBoosting.create(n_stat, n_act, n_learner, args.genes, args.eps_period, args.mu_scale, args.sigma, dqn=args.dqn)

    env = Pong(model, render=False)
    for i in xrange(n_eps):
        env.execute()
        if i%(one_gen*20) == 0:
            model.save(args.prefix+
                    '_'+isDqn+
                    '_'+str(args.mu_scale)+
                    '_'+str(args.sigma)+
                    '_'+str(args.genes)+
                    '_'+str(args.eps_period)+
                    '_'+str(i/one_gen)+'.pkl')
    model.save(args.prefix+
            '_'+isDqn+
            '_'+str(args.mu_scale)+
            '_'+str(args.sigma)+
            '_'+str(args.genes)+
            '_'+str(args.eps_period)+
            '_'+str(i/one_gen)+'.pkl')
