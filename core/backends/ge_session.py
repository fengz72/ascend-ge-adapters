"""GE Session compiles the exported graph when loading it."""


def build(graph, io, options, target, directory):
    if graph["format"] not in ("air", "onnx"):
        raise ValueError("GE Session requires AIR or ONNX")
    return None
