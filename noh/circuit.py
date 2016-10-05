from noh.component import Component

class Collection(object):
    keys = []
    values = []

    def __init__(self, collection):
        if isinstance(collection, dict):
            for key, value in collection.items():
                self.keys.append(key)
                self.values.append(key)

        if isinstance(collection, list):
            keys = [item.__class__.__name__.lower() for item in collection]

            counts = {}

            for key, value in zip(keys, collection):
                if keys.count(key) > 1:
                    if key not in counts:
                        counts[key] = 0
                    counts[key] += 1
                    key = '{}{}'.format(key, counts[key])
                self.keys.append(key)
                self.values.append(value)

    def __getitem__(self, key):
        if isinstance(key, int):
            index = key
        else:
            index = self.keys.index(key)
        return self.values[index]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            index = key
        else:
            index = self.keys.index(key)
        self.values[index] = value

    def __delitem__(self, key):
        if isinstance(key, int):
            index = key
        else:
            index = self.keys.index(key)
        del self.keys[key]
        del self.values[key]

    def __iter__(self):
        return iter(self.keys)

    def __getslice__(self, i, j):
        raise NotImplementedError("To be implemented")

    def __setslice__(self, i, j, values):
        raise NotImplementedError("To be implemented")

    def __delslice__(self, i, j):
        raise NotImplementedError("To be implemented")

    def __getattr__(self, key):
        return self.__getitem__(key)


class PropRule(Collection):
    def __init__(self, components):
        super(PropRule, self).__init__(components)
        self.components = components

    def __call__(self, data):
        raise NotImplementedError("`__call__` must be explicitly overridden")


class TrainRule(Collection):
    def __init__(self, components):
        super(TrainRule, self).__init__(components)
        self.components = components

    def __call__(self, data, label, epoch):
        raise NotImplementedError("`__call__` must be explicitly overridden")


class Planner(object):
    def __init__(self, components, rule_dict={}, default_prop=None, default_train=None):
        self.components = components
        self.rules = {}
        if default_prop is not None:
            self.rules["prop"] = default_prop(components)
            self.prop_rule = self.rules['prop']
        if default_train is not None:
            self.rules["train"] = default_train(components)
            self.train_rule = self.rules['train']

        for name in rule_dict:
            Rule = rule_dict[name]
            self.rules[name] = Rule(components)

    def set_prop(self, name):
        self.prop_rule = self.rules[name]

    def set_train(self, name):
        self.train_rule = self.rules[name]

    def __call__(self, data):
        return self.prop_rule(data)

    def train(self, data, label, epoch):
        return self.train_rule(data, label, epoch)


class Circuit(Collection, Component):
    def __init__(self, PlannerClass, components, rule_dict,
                 default_prop=None, default_train=None):
        super(Circuit, self).__init__(components)
        self.planner = PlannerClass(components, rule_dict,
                                    default_prop, default_train)

    def __call__(self, data, **kwargs):
        return self.planner(data)

    def train(self, data, label, epochs):
        return self.planner.train(data, label, epochs)

    def __getattr__(self, key):
        return self.planner.rules[key]