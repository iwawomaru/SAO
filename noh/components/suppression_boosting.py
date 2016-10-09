from noh import Circuit
from noh.circuit import PropRule
from noh.components import Random, Const, DQN
import numpy as np


class SimpleProp(PropRule):
    component_id_list = range(100)

    def __init__(self, components):
        super(SimpleProp, self).__init__(components)
        self.id = self.component_id_list.pop(0)

    def __call__(self, data):
        return self.components[self.id](data)


class EachProp(PropRule):

    def __init__(self, components):
        super(EachProp, self).__init__(components)

    def __call__(self, data):
        return [c(data) for c in self.components]


class LearnerSet(Circuit):

    def __init__(self, components, RuleClassDict):
        super(LearnerSet, self).__init__(components, RuleClassDict)
        self.n_components = len(components)
        self.f_go = False

    @classmethod
    def create(cls, n_stat, n_act, n_learner):
        component_list = [Random(n_input=n_stat, n_output=n_act)] + \
                         [Const(n_input=n_stat, n_output=n_act, const_output=n) for n in xrange(1, n_learner)]
        PropRulesDict = {"prop"+str(i): SimpleProp for i in xrange(n_learner)}
        return LearnerSet(component_list, PropRulesDict)


class GeneSet(LearnerSet):

    def __init__(self, components, RuleClassDict):
        super(GeneSet, self).__init__(components, RuleClassDict)

    @classmethod
    def create(cls, n_stat, n_act, n_learner, n_gene):
        component_list = [PropLearner({'learner_set':LearnerSet.create(n_stat, n_act, n_learner)}) for n in xrange(n_gene)]
        PropRulesDict = {"gene"+str(i): SimpleProp for i in xrange(n_gene)}
        return GeneSet(component_list, PropRulesDict)


class PropLearner(PropRule):

    name_list = []
    threshold = -1.
    def __init__(self, components):
        super(PropLearner, self).__init__(components)
        self.reward = 0.
        self.evidence = 0.
        self.gene = {name: {'sigma':np.random.rand(), 'mu':np.random.rand()} for name in self.name_list}

    def __call__(self, data):
        if not self.components["learner_set"].f_go:
            self.prop = np.random.choice(self.name_list)
            self.components["learner_set"].f_go = True
            self.components["learner_set"].set_default_prop(name=self.prop)

        self.evidence += self.reward * self.sample_normal(self.prop)
        res = self.components["learner_set"](data)

        """ kashikoku shitai here """
        if self.evidence < self.threshold:
            self.components["learner_set"].f_go = False
            self.evidence = 0.
            print "accumulation"
        return res

    def sample_normal(self, prop):
        param = self.gene[prop]
        return np.random.normal(loc=param['mu'], scale=param['sigma'])

    def set_reward(self, reward):
        #print "called PropLearner's set_reward"
        self.reward = reward

    def reset(self):
        #print "called PropLearner's reset"
        self.evidence = 0.



class GALearner(PropRule):

    name_list = []
    n_gene = 5
    eps_period = 5
    mutation_rate = 0.1
    threshold = -1.
    def __init__(self, components):
        super(GALearner, self).__init__(components)
        self.reward = 0.
        self.reward_sum = 0.
        self.reward_sum_history = []
        self.evidence = 0.
        self.eps = 0
        self.genes = [{name: {'sigma':np.random.rand(), 'mu':np.random.rand()} for name in self.name_list} for i in xrange(self.n_gene)]

    def __call__(self, data):
        if not self.components["learner_set"].f_go:
            self.prop = np.random.choice(self.name_list)
            self.components["learner_set"].f_go = True
            self.components["learner_set"].set_default_prop(name=self.prop)

        gene_id = self.eps % self.eps_period / self.eps_period
        self.evidence += self.reward * self.sample_normal(gene_id, self.prop)
        res = self.components["learner_set"](data)

        """ kashikoku shitai here """
        if self.evidence < self.threshold:
            self.components["learner_set"].f_go = False
            self.evidence = 0.
            #print "accumulation"
        return res

    def sample_normal(self, gene_id, prop):
        param = self.genes[gene_id][prop]
        return np.random.normal(loc=param['mu'], scale=param['sigma'])

    def alternate(self):
        print "call alternate"
        # selection
        n_survive_gene = len(self.reward_sum_history) / 2 + 1
        new_genes = []
        sorted_reward = sorted(self.reward_sum_history)
        for sr in sorted_reward[:n_survive_gene]:
            idx = self.reward_sum_history.index(sr)
            new_genes.append(self.genes[idx])

        # crossing
        for i in xrange(self.n_gene):
            u = np.random.choice(new_genes)
            v = np.random.choice(new_genes)
            for n in self.name_list:
                g = np.random.choice([u, v])
                self.genes[i][n] = g[n]
                # mutation
                for k in g[n].keys():
                    if np.random.rand() < self.mutation_rate:
                        self.genes[i][n][k] += self.genes[i][n][k] * (np.random.rand()-0.5)

    def set_reward(self, reward):
        #print "called GALearner's set_reward"
        self.reward = reward
        self.reward_sum += reward

    def reset(self):
        #print "called GALearner's reset"
        self.evidence = 0.

        # change gene
        if self.eps != 0 and self.eps%self.eps_period == 0:
            self.reward_sum_history.append(self.reward_sum)
            self.reward_sum = 0.

        # change generation
        if self.eps != 0 and self.eps%(self.eps_period*self.n_gene) == 0:
            self.alternate()
            self.reward_sum_history = []

        self.eps += 1


class SuppressionBoosting(Circuit):
    def __init__(self, components, RuleClassDict):
        super(SuppressionBoosting, self).__init__(components, RuleClassDict, default_prop_name="ga_learner")
        self.f_go = False

    @classmethod
    def create(cls, n_stat, n_act, n_learner):
        components = {"learner_set": LearnerSet.create(n_stat, n_act, n_learner),
                      "suppressor": None}
        PropLearner.name_list = components["learner_set"].rules.keys()
        return SuppressionBoosting(components, {"prop_learner": PropLearner})

    def stop(self):
        self.f_go = False


class GASuppressionBoosting(Circuit):
    def __init__(self, components, RuleClassDict):
        super(GASuppressionBoosting, self).__init__(components, RuleClassDict, default_prop_name="ga_learner")
        self.f_go = False

    @classmethod
    def create(cls, n_stat, n_act, n_learner, n_gene, eps_period):
        components = {"learner_set": LearnerSet.create(n_stat, n_act, n_learner),
                      "suppressor": None}
        GALearner.name_list = components["learner_set"].rules.keys()
        GALearner.n_gene = n_gene
        GALearner.eps_period = eps_period
        return GASuppressionBoosting(components, {"ga_learner": GALearner})

    def stop(self):
        self.f_go = False

