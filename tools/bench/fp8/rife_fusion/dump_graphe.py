#!/usr/bin/env python3
"""Liste le graphe ONNX de RIFE (noeuds, entrees, sorties, attributs) pour reperer les coupes."""
import sys
from collections import Counter

import onnx

m = onnx.load(sys.argv[1])
g = m.graph
print(len(g.node), [i.name for i in g.input], [o.name for o in g.output])
print(Counter(n.op_type for n in g.node))
for n in g.node:
    attrs = [(a.name, onnx.helper.get_attribute_value(a)) for a in n.attribute if a.name != "value"]
    print(n.name, n.op_type, list(n.input), "->", list(n.output), attrs[:5])
