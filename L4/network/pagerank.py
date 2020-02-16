import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def showGraph(weights_edges):
    G = nx.DiGraph()
    G.add_weighted_edges_from(weights_edges)
    layout = nx.spring_layout(G)
    nx.draw(G, pos=layout, with_labels=True, hold=False)
    plt.show()


def get_weight(adjmatrix: np.array):
    nodes = adjmatrix.shape[0]
    rows, cols = adjmatrix.shape
    weights_edges = []
    for i in range(rows):
        for j in range(cols):
            if adjmatrix[i][j] != 0:
                weights_edges.append((i, j, adjmatrix[i][j]))
    return weights_edges


def check(adjmatrix: np.array):
    list1 = list(np.sum(adjmatrix, axis=0))
    list2 = list(np.sum(adjmatrix, axis=1))
    if 0.0 in list1 or 0.0 in list2:
        return False
    return True


def work(a, w):
    for i in range(100):
        w = np.dot(a, w)
        print(w)


def random_work(a, w, n):
    d = 0.85
    for i in range(100):
        w = (1 - d) / n + d * np.dot(a, w)
        print(w)


# Rank Sink（等级沉没）一个网页只有出链，没有入链。比如节点A迭代下来，会导致这个网页的PR值为0
# Rank Leak（等级泄露）一个网页没有出链，就像是一个黑洞一样，吸收了别人的影响力而不释放，最终会导致其他网页的PR值为0


if __name__ == "__main__":
    a = np.array([
        [0, 1 / 2, 1, 0],
        [1 / 3, 0, 0, 1 / 2],
        [1 / 3, 0, 0, 1 / 2],
        [1 / 3, 1 / 2, 0, 0]
    ]
    )

    # node id =0 入度为0
    a_leak = np.array([[0, 0, 0, 1 / 2],
                       [0, 0, 0, 1 / 2],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])

    # node id =0 出度为0
    a_sink = np.array([[0, 0, 0, 0],
                       [1 / 2, 0, 0, 1],
                       [0, 1, 1, 0],
                       [1 / 2, 0, 0, 0]])

    testALlist = [a, a_sink, a_leak]
    for testA in testALlist:
        weights_edges = get_weight(testA)
        showGraph(weights_edges)
        b = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
        w = b
        np.set_printoptions(suppress=True)
        work(testA, w)
        print("#" * 100)

    print("random_work -----------------------")

    testALlist = [a, a_sink, a_leak]
    for testA in testALlist:
        weights_edges = get_weight(testA)
        showGraph(weights_edges)
        b = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
        w = b
        n = testA.shape[0]
        np.set_printoptions(suppress=True)
        random_work(testA, w, n)
        print("#" * 100)


