


class Signature(object):

    def __init__(self, genes):
        self.genes = genes
        self.accuracies = []
        self.weight = 0 # how many times this signature was selected
        self.independentAccuracies = []
        self.id = str(sorted(self.genes))

    def __eq__(self, other):               
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
        
    def __str__(self):
        return self.id

    def getN(self):
        return len(self.genes)

    def getMaxAcc(self):
        return max(self.accuracies)

    def getMinAcc(self):
        return min(self.accuracies)
    
    def getAccuracies(self):
        return self.accuracies

    def addAccuracy(self, acc):
        self.accuracies.append(acc)        
    
    def addIndependentTestAccuracy(self, acc):
        self.independentAccuracies.append(acc)

    def getIndependentTestAccuracies(self):
        return self.independentAccuracies

    def toGeneNamesList(self, names):
        return [names[gene] for gene in self.genes]
        