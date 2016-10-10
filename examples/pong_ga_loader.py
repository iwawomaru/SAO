import sys, os, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import Pong
from noh.components import GASuppressionBoosting


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GA model loader')
    parser.add_argument('model', type=str, help='Model file path')
    parser.add_argument('--mu_scale', '-m', default=0.6, type=float, help='Scale of mu')
    parser.add_argument('--sigma', '-s', default=0.001, type=float, help='Sigma')
    parser.add_argument('--genes', '-g', default=8, type=int, help='# of genes')
    parser.add_argument('--eps_period', '-e', default=5, type=int, help='episode period of genes')
    parser.add_argument('--dqn', action='store_true', default=False, help='DQN mode flag')
    args = parser.parse_args()

    n_stat = Pong.n_stat
    n_act = Pong.n_act
    n_learner = 4
    n_gen = 10
    one_gen = args.genes * args.eps_period
    n_eps = one_gen * n_gen

    n_stat = Pong.n_stat
    n_act = Pong.n_act
    n_learner = 4
    n_gen = 100
    one_gen = args.genes * args.eps_period
    n_eps = one_gen * n_gen
    isDqn = 'dqn' if args.dqn else 'random'

    model = GASuppressionBoosting.create(n_stat, n_act, n_learner, args.genes, args.eps_period, args.mu_scale, args.sigma, dqn=args.dqn)
    model.load(args.model)

    print "model parameter: "
    for n in model.rules["ga_learner"].name_list:
        print n+": ", model.rules["ga_learner"].genes[0][n]

    env = Pong(model, render=True)
    while True:
        env.execute()
        #if i%(one_gen*2) == 0:
            #model.save('gasb_'+str(i/one_gen)+'gen.pkl')
    #model.save('gasb_'+str(n_gen)+'gen.pkl')
