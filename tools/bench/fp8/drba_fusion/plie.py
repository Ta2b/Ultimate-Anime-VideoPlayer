"""Plie les constantes du graphe DistilDRBA (Shape -> constante, noeuds a entrees constantes evalues) et ecrit l'ONNX plie."""
import sys, numpy as np, onnx
from onnx import numpy_helper, shape_inference, helper
from onnx.reference import ReferenceEvaluator
m = onnx.load(sys.argv[1]); m = shape_inference.infer_shapes(m)
g = m.graph
shapes = {}
for vi in list(g.value_info)+list(g.input)+list(g.output):
    shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
const = {i.name: numpy_helper.to_array(i) for i in g.initializer}
for n in g.node:
    if n.op_type == "Constant":
        const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
new = []
folded = 0
for n in g.node:
    if n.op_type == "Constant": continue
    if n.op_type == "Shape" and n.input[0] in shapes and all(shapes[n.input[0]]):
        const[n.output[0]] = np.array(shapes[n.input[0]], np.int64); folded += 1; continue
    if n.input and all((i in const) or i=="" for i in n.input) and n.op_type not in ("Conv",):
        try:
            r = ReferenceEvaluator(helper.make_model(helper.make_graph([n], "t", [], []), opset_imports=m.opset_import) if False else None) if False else None
        except Exception: r=None
        # evaluation par un mini modele
        ins = [helper.make_tensor_value_info(i, onnx.TensorProto.FLOAT, None) for i in []]
        mm = helper.make_model(helper.make_graph([n], "t", [helper.make_tensor_value_info(i, numpy_helper.from_array(const[i]).data_type, list(const[i].shape)) for i in n.input if i],
              [helper.make_tensor_value_info(o, 0, None) for o in n.output]), opset_imports=m.opset_import)
        try:
            outs = ReferenceEvaluator(mm).run(None, {i: const[i] for i in n.input if i})
            for o, v in zip(n.output, outs): const[o] = np.asarray(v)
            folded += 1; continue
        except Exception as e:
            print("echec pli", n.name, n.op_type, e, file=sys.stderr)
    new.append(n)
print("plies", folded, "restants", len(new), file=sys.stderr)
used = set()
for n in new: used.update(i for i in n.input if i)
used.update(o.name for o in g.output)
del g.node[:]; g.node.extend(new)
del g.initializer[:]
for k in used:
    if k in const and k not in {i.name for i in g.input}:
        g.initializer.append(numpy_helper.from_array(const[k], k))
del g.value_info[:]
m = shape_inference.infer_shapes(m)
onnx.save(m, sys.argv[2])
