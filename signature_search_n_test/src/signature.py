
import numpy as np
from pandas import DataFrame

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

    def addMethod(self, method):
        self.methods.add(method)
    
    def getMethods(self):
        return self.methods

    def setScore(self, estimator_name, score):
        self.scores[estimator_name] = score    

    def hasScore(self, estimator_name):
        if estimator_name in self.scores:
            return True
        return False

    def getScore(self, estimator_name):
        return self.scores[estimator_name]

    def addIndependentScore(self, estimator_name, score, y_pred, y_true):       
        if estimator_name not in self.independent_scores:
            self.independent_scores[estimator_name] = {}

        self.independent_scores[estimator_name]['score'] = score
        self.independent_scores[estimator_name]['y_pred'] = y_pred
        self.independent_scores[estimator_name]['y_true'] = y_true

    def getMaxScorePairs(self):
        max_score = -np.inf
        pairs = []
        for estimator_name in self.scores:           
            score = self.scores[estimator_name]
            independent_score =  -5.0
            if estimator_name in self.independent_scores:
                independent_score = self.independent_scores[estimator_name]['score']

            if score > max_score:
                max_score = score
                pairs = [(max_score, independent_score, self.size(), self.methods, estimator_name, self)]
            elif score == max_score:
                pairs.append((max_score, independent_score, self.size(), self.methods, estimator_name, self))
        return pairs

class Signatures(object):

    def __init__(self):
        self.signatures = {}

    def get(self, genes):
        signature = Signature(genes)
        if signature not in self.signatures:
            self.signatures[signature] = signature
        return self.signatures[signature]

    def getSignaturesMaxScore(self, delta):
        pairs = []
        for sig in self.signatures.values():
            for pair in sig.getMaxScorePairs():
                pairs.append(pair)            

        pairs = sorted(pairs, reverse=True)
        selected_pairs = [pair for pair in pairs if pair[0] > pairs[0][0]-delta]
        return sorted(selected_pairs, key=lambda tup: tup[2]) #ordered by size       

    def save(self, filename):

        estimator_names = set()
        for signature in self.signatures:
            for name in signature.scores:
                estimator_names.add(name)
        estimator_names = sorted(list(estimator_names))

        header = ['size']
        for name in estimator_names:
            header.append('cv_'+name)
        header.append('proteins')
        header.append('methods (not verified in other ranks)')

        matrix = []
        count = 1
        for signature in self.signatures:
            row = [signature.size()]           
            for estimator_name in estimator_names:
                if signature.hasScore(estimator_name):
                    row.append(signature.getScore(estimator_name))
                else:
                    row.append(np.nan)
            count+=1
            row.append(str(signature.genes))
            row.append(str(list(signature.methods)))
            matrix.append(row)
        
        df = DataFrame(data=matrix, columns=header)
        df.sort_values([df.columns[1],df.columns[0]], ascending=[0,1])
        df.to_csv(filename, header=True)