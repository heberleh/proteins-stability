

class PossibleEdge(object):

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.count = 1
    
    def __eq__(self, other):
        if self.source == other.source and self.target == other.target:
            return True
        elif self.source == other.target and self.target == other.source:
            return True
        else:
            return False

    def increment_count(self):
        self.count += 1

        
           