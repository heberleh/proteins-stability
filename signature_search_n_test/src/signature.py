


class Signature(object):

    def __init__(self, genes):
        self.genes = genes
        self.accuracies = []
        self.weight = 0 # how many times this signature was selected
        self.independentAccuracies = []

    
    def __eq__(self, other):
        if (len(self.genes)!=len(other.genes)):
            return False
                
        for gene1 in self.genes:
            if gene1 not in other.genes:
                return False

        return True

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