from numpy import matrix, isnan, array
from scipy.stats import pearsonr


class CoExpressionGraph(object):

    def __init__(self, x, cut_off):
        """
        :param x: matrix
        :param y: labels/classes
        :param cut_off: value in the range [-1,1], if correlation > cutoff -> create edge
        :return: graph dictionary as  {1:[2,3,4], 3:[3,2]}
        """
        self.edges = {}   # {1:[2,3,4], 3:[3,2]}
        self.create(matrix(x), cut_off)

    def create(self, x, cut_off):
        m = len(x)
        n = len(x[0])
        if m > n:  # there is always more genes than samples (in 2015s)
            x = x.transpose()

        for node in range(len(x)):
            self.edges[node] = set()

        for i in range(len(x)):
            for j in range(i+1, len(x)):
                r, p = pearsonr(array(x[i])[0], array(x[j])[0])
                if not isnan(p) and p > cut_off:
                    self.create_edge(i, j)

    def create_edge(self, n0, n1):
        self.edges[n0].add(n1)
        self.edges[n1].add(n0)

    def print_graph(self):
        txt = ""
        for node in self.edges:
            txt += str(self.edges[node])+"\n"
        print txt

    def get_graph(self):
        return self.edges
