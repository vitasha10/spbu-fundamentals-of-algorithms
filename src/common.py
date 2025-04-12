from collections import namedtuple
from typing import Union

import numpy as np
from numpy.typing import NDArray
import networkx as nx


ProblemCase = namedtuple("ProblemCase", "input, output")
NDArrayInt = NDArray[np.int_]
NDArrayFloat = NDArray[np.float_]
AnyNxGraph = Union[nx.Graph, nx.DiGraph]

