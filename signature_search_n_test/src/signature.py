
import numpy as np

class Signature(object):

    def __init__(self, genes):
        self.genes = genes
        self.weight = 0 # how many times this signature was selected        
        self.id = str(sorted(self.genes))

        self.scores = {}
        self.methods = set()
        self.independent_scores = {}

    def __eq__(self, other):               
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
        
    def __str__(self):
        return self.id

    def size(self):
        return len(self.genes)
        
    def addScore(self, method, estimator_name, score):
        if estimator_name in self.scores:            
            self.methods.add(method)
        else:
            self.scores[estimator_name] = score
            self.methods.add(method)       

    def hasScore(self, estimator_name):
        if estimator_name in self.scores:
            return True
        return False

    def getScore(self, estimator_name):
        return self.scores[estimator_name]

    def addIndependentScore(self, estimator_name, score, y_pred):       
        if estimator_name not in self.independent_scores:
            self.independent_scores[estimator_name] = {}

        self.independent_scores[estimator_name]['score'] = score
        self.independent_scores[estimator_name]['y_pred'] = y_pred

    def getMaxScorePairs(self):
        max_score = -np.inf
        pairs = []
        for estimator_name in self.scores:           
            score = self.scores[estimator_name]
            if score > max_score:
                max_score = score
                pairs = [(max_score, self.independent_scores[estimator_name]['score'], self.size(), self.methods, estimator_name, self)]
            elif score == max_score:
                pairs.append((max_score, self.independent_scores[estimator_name]['score'], self.size(), self.methods, estimator_name, self))
        return pairs

class Signatures(object):

    def __init__(self):
        self.signatures = {}

    def get(self, genes):
        signature = Signature(genes)
        if signature not in self.signatures:
            self.signatures[signature] = signature
        return self.signatures[signature]

    def getSignaturesMaxScore(self):
        pairs = []
        for sig in self.signatures.values():
            for pair in sig.getMaxScorePairs():
                pairs.append(pair)

        pairs = sorted(pairs, reverse=True)
        selected_pairs = [pair for pair in pairs if pair[0] > pairs[0][0]-0.009]
        return sorted(selected_pairs, key=lambda tup: tup[2]) #ordered by size       
