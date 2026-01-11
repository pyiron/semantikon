# semantikon representation

The semantikon representation of a workflow is a **nested dictionary** with the following entries:

- `label` (required): Name of the workflow
- `inputs` (required): Input arguments and their metadata (see below)
- `outputs` (required): Output arguments and their metadata (see below)
- `nodes` (required): Atomic nodes or nested workflows (see below)
- `type` (required): It must be "Workflow" (otherwise "Function" for the atomic nodes)
- `function` (optional): The underlying python function and its metadata (see below)

The input and output arguments must be given by a dictionary whose key is the argument name in the case of the input, and a label in the case of the output. The value must be given by a dictionary, which can in principle contain anything (or nothing). Following arguments can be currently understood by `semantikon`:

- `value`: Literal value of the argument
- `dtype`: Data type; currently this has no meaning for `semantikon` except for data classes
- `uri`: URIRef of what the argument represents (must be a class)
- `triples`: A double, a triple or a list/tuple of doubles or triples. See below for the syntax
- `restrictions`: owl-type restrictions. See below for the syntax
- `units`: QUDT URI or str that can be understood by pint (and will be later translated to pint URI)
- `derived_from`: To indicate that (typically) an output is derived from an input. For example painting a car where the output is basically the same car as the input with a different color.

An atomic node is represented by a dictionary, containing the following arguments:

- `label` (required): Name of the node
- `inputs` (required): Input arguments and their metadata
- `outputs` (required): Output arguments and their metadata
- `type` (required): "Function" for an atomic node
- `function` (required): Underlying python function (note: **required** for an atomic node)

# Knowledge graph
