


class Signature(object):

    def __init__(self, genes):
        self.genes = genes
        self.accuracies = []
        self.predicted = []
        self.truth = []
    
    def __eq__(self, other):
        if (len(self.genes)!=len(other.genes)):
            return False
                
        for gene1 in self.genes:
            if gene1 not in other.genes:
                return False

        return True

    def get_n(self):
        return len(self.genes)

    def add_pair_truth_prediced(self, t, p):
        self.truth.append(t)
        self.predicted.append(p)

    def get_truth(self):
        return self.truth
    
    def get_predicted(self):
        return self.predicted
