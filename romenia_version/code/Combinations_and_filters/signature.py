


class Signature(object):

    def __init__(self, genes):
        self.genes = genes
        self.accuracies = []
        self.weight = 0
        self.independent_accs = []

    
    def __eq__(self, other):
        if (len(self.genes)!=len(other.genes)):
            return False
                
        for gene1 in self.genes:
            if gene1 not in other.genes:
                return False

        return True

    def get_n(self):
        return len(self.genes)

    def get_max_acc(self):
        return max(self.accuracies)

    def get_min_acc(self):
        return min(self.accuracies)
    
    def get_accs(self):
        return self.accuracies

    def get_weight(self):
        return self.weight

    def add_acc(self, acc):
        self.accuracies.append(acc)
        self.weight += 1
    
    def add_independent_test_acc(self, acc):
        self.independent_accs.append(acc)

    def get_independent_test_accs(self):
        return self.independent_accs