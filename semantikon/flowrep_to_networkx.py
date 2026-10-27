from __future__ import annotations

import copy
import json
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import cached_property
from hashlib import sha256
from typing import Annotated, Any, cast, get_args, get_origin

import flowrep as fr
import networkx as nx
from flowrep.prospective import transformers
from pyiron_snippets import retrieve
from rdflib import BNode
from rdflib.term import IdentifiedNode

from semantikon.converter import get_function_dict, parse_metadata
from semantikon.datastructure import TypeMetadata
from semantikon.graph_nodes import IO, Input, Node, Output


def annotation_to_type_hint(annotation: Any) -> Any | None:
    """
    Extract the underlying type hint from a parameter annotation.

    Plain annotations (``int`` in ``def foo(x: int)``) are returned as-is;
    ``Annotated[T, ...]`` is unwrapped to ``T``.

    Args:
        annotation: A parameter annotation, plain or ``Annotated``.

    Returns:
        The underlying type, or ``None`` if ``annotation`` is ``None``.
    """
    if annotation is None:
        return None
    elif get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    else:
        return annotation


def annotation_to_type_metadata(annotation: Any) -> TypeMetadata | None:
    """
    Extract ``TypeMetadata`` from an ``Annotated`` parameter annotation.

    Plain annotations (``int`` in ``def foo(x: int)``) carry no metadata and
    yield ``None``; only ``Annotated[T, ...]`` is parsed.

    Args:
        annotation: A parameter annotation, plain or ``Annotated``.

    Returns:
        The parsed metadata, or ``None`` if ``annotation`` is not ``Annotated``.
    """
    if annotation is None or get_origin(annotation) is not Annotated:
        return None
    return parse_metadata(annotation)


@dataclass(frozen=True, slots=True)
class ANode(Node):
    owner: ANode | None = None
    counter: int = 0

    def __str__(self) -> str:
        base = f"{self.owner}-{self.name}" if self.owner else self.name
        return f"{base}_{self.counter}"


@dataclass(frozen=True, slots=True)
class AIO(IO):
    node: ANode
    counter: int = 0


@dataclass(frozen=True, slots=True)
class AInput(AIO):
    def __str__(self) -> str:
        return f"{self.node}-inputs-{self.port}_{self.counter}"


@dataclass(frozen=True, slots=True)
class AOutput(AIO):
    def __str__(self) -> str:
        return f"{self.node}-outputs-{self.port}_{self.counter}"


@dataclass(frozen=True, slots=True)
class TNodeData:
    type: str | None = None
    identifier: str | None = None
    label: str | None = None
    function: dict | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        if self.type is not None:
            attrs["type"] = self.type
        if self.identifier is not None:
            attrs["identifier"] = self.identifier
        if self.label is not None:
            attrs["label"] = self.label
        if self.function is not None:
            attrs["function"] = self.function
        attrs.update(self.extras)
        return attrs


@dataclass(frozen=True, slots=True)
class TIOData:
    position: int
    dtype: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "position": self.position,
        }
        if self.dtype is not None:
            attrs["dtype"] = self.dtype
        attrs.update(self.metadata)
        return attrs


@dataclass(frozen=True, slots=True)
class TInputData(TIOData):
    default: Any | None = None
    has_default: bool = False

    def to_attrs(self) -> dict[str, Any]:
        attrs = TIOData.to_attrs(self)
        if self.has_default:
            attrs["default"] = self.default
        return attrs


@dataclass(frozen=True, slots=True)
class TOutputData(TIOData):
    pass


@dataclass(frozen=True, slots=True)
class ANodeData:
    """A-Box node attributes: a reference to the recipe node plus the status."""

    recipe: Node
    status: str = "pending"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        return {"recipe": self.recipe, "status": self.status, **self.extras}


@dataclass(frozen=True, slots=True)
class AIOData:
    """A-Box port attributes: a reference to the recipe port plus the value."""

    recipe: IO
    value: Any | None = None
    has_value: bool = False
    status: str = "missing"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {"recipe": self.recipe, "status": self.status}
        if self.has_value:
            attrs["value"] = self.value
        attrs.update(self.extras)
        return attrs


class SemantikonRecipeGraph(nx.DiGraph):
    """Workflow recipe (T-Box) graph with deterministic namespace fragments.

    The graph only stores suffix fragments (e.g. ``abc123_``) for type-level
    identifiers. Ontology-specific base namespaces are applied later by the
    ontology serialization layer. It holds no values; see
    :class:`SemantikonInstanceGraph` for the instance (A-Box) counterpart.
    """

    @property
    def recipe_graph(self) -> SemantikonRecipeGraph:
        return self

    def recipe_of(self, node: Node | IO) -> Node | IO:
        return node

    def get_data(self, node: Node | IO) -> dict[str, Any]:
        return dict(self.nodes[node])

    def _validate_semantikon_attrs(
        self, step: IO | Node, attrs: dict[str, Any]
    ) -> dict[str, Any]:
        if "value" in attrs:
            raise ValueError(
                "Recipe graphs do not hold values; use SemantikonInstanceGraph."
            )

        if isinstance(step, Node):
            known = {"type", "identifier", "label", "function"}
            node_meta = TNodeData(
                type=attrs.get("type"),
                identifier=attrs.get("identifier"),
                label=attrs.get("label"),
                function=attrs.get("function"),
                extras={k: v for k, v in attrs.items() if k not in known},
            )
            return node_meta.to_attrs()

        if isinstance(step, Input):
            known = {"position", "dtype", "default"}
            input_meta = TInputData(
                position=attrs["position"],
                dtype=attrs.get("dtype"),
                default=attrs.get("default"),
                has_default="default" in attrs,
                metadata={k: v for k, v in attrs.items() if k not in known},
            )
            return input_meta.to_attrs()

        if isinstance(step, Output):
            known = {"position", "dtype"}
            output_meta = TOutputData(
                position=attrs["position"],
                dtype=attrs.get("dtype"),
                metadata={k: v for k, v in attrs.items() if k not in known},
            )
            return output_meta.to_attrs()

        raise TypeError(f"Unknown step type: {type(step)}")

    def add_node(self, node_for_adding, **attr):  # type: ignore[override]
        assert isinstance(node_for_adding, (Node, IO))
        normalized_attr = self._validate_semantikon_attrs(node_for_adding, attr)
        super().add_node(node_for_adding, **normalized_attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        assert all(isinstance(n, (Node, IO, tuple)) for n in nodes_for_adding)
        for n in nodes_for_adding:
            if isinstance(n, tuple):
                node, node_attr = n
                normalized_attr = self._validate_semantikon_attrs(
                    node, attr | node_attr
                )
                super().add_node(node, **normalized_attr)
            else:
                normalized_attr = self._validate_semantikon_attrs(n, attr)
                super().add_node(n, **normalized_attr)

    @cached_property
    def t_ns(self) -> str:
        """Type-level namespace fragment for this graph."""
        h = (
            "W" + _get_graph_hash(self, with_global_inputs=False)[:8]
            if self.graph["prefix"] is None
            else self.graph["prefix"]
        )
        return h + "_"

    def _get_data_node(self, io: IO) -> str:
        while True:
            candidate = [c for c in self.predecessors(io) if isinstance(c, IO)]
            assert len(candidate) <= 1
            if len(candidate) == 0:
                return f"{io}_data"
            io = candidate[0]

    def _initialize_type(self):
        owners = {n.owner for n in self.nodes if isinstance(n, Node) and n.owner}
        for node, data in self.nodes.data():
            if isinstance(node, Node) and "type" not in data:
                data["type"] = "workflow" if node in owners else "atomic"

    def get_type(self, node_name: Node) -> str:
        """
        Get the type of a node in the graph.

        Parameters:
            node_name (Node): The name of the node for which to retrieve the
                type.

        Returns:
            str: The type of the node. Possible values are "atomic",
                "constant", or "workflow".

        Raises:
            ValueError: If the node is not found in the graph.
        """
        if "type" not in self.nodes[node_name]:
            self._initialize_type()
        return self.nodes[node_name]["type"]


SemantikonDiGraph = SemantikonRecipeGraph


class SemantikonInstanceGraph(nx.DiGraph):
    """Workflow instance (A-Box) graph.

    Nodes are :class:`ANode`, :class:`AInput` and :class:`AOutput`. Each node
    stores a ``recipe`` attribute referencing the corresponding node of the
    :class:`SemantikonRecipeGraph` (``recipe_graph``), together with the
    instance-specific data (``value``, ``status``, ``hash``).
    """

    def __init__(
        self, *args, recipe_graph: SemantikonRecipeGraph | None = None, **attr
    ):
        super().__init__(*args, **attr)
        if recipe_graph is not None:
            self.graph["recipe_graph"] = recipe_graph

    @property
    def recipe_graph(self) -> SemantikonRecipeGraph:
        return self.graph["recipe_graph"]

    def recipe_of(self, node: ANode | AIO) -> Node | IO:
        return self.nodes[node]["recipe"]

    def get_data(self, node: ANode | AIO) -> dict[str, Any]:
        """Recipe attributes of the node updated with its instance attributes."""
        return self.recipe_graph.get_data(self.recipe_of(node)) | dict(self.nodes[node])

    def _validate_instance_attrs(
        self, step: ANode | AIO, attrs: dict[str, Any]
    ) -> dict[str, Any]:
        if "recipe" not in attrs:
            raise ValueError(f"{step} requires a reference to its recipe node.")
        recipe = attrs["recipe"]
        recipe_type: type[Node | Input | Output]
        if isinstance(step, ANode):
            recipe_type = Node
        elif isinstance(step, AInput):
            recipe_type = Input
        elif isinstance(step, AOutput):
            recipe_type = Output
        else:
            raise TypeError(f"Unknown step type: {type(step)}")
        if type(recipe) is not recipe_type:
            raise ValueError(f"{step} requires a {recipe_type.__name__} recipe node.")
        if "recipe_graph" not in self.graph or recipe not in self.recipe_graph:
            raise ValueError(
                f"{step} references a recipe node not in its recipe graph."
            )
        known = {"recipe", "status", "value"}
        extras = {k: v for k, v in attrs.items() if k not in known}
        if isinstance(step, ANode):
            return ANodeData(
                recipe=cast(Node, recipe),
                status=attrs.get("status", "pending"),
                extras=extras,
            ).to_attrs()
        if isinstance(step, AIO):
            has_value = "value" in attrs
            return AIOData(
                recipe=cast(IO, recipe),
                value=attrs.get("value"),
                has_value=has_value,
                status=attrs.get("status", "ready" if has_value else "missing"),
                extras=extras,
            ).to_attrs()
        raise TypeError(f"Unknown step type: {type(step)}")

    def add_node(self, node_for_adding, **attr):  # type: ignore[override]
        assert isinstance(node_for_adding, (ANode, AIO))
        super().add_node(
            node_for_adding, **self._validate_instance_attrs(node_for_adding, attr)
        )

    def add_nodes_from(self, nodes_for_adding, **attr):
        for n in nodes_for_adding:
            if isinstance(n, tuple):
                self.add_node(n[0], **(attr | n[1]))
            else:
                self.add_node(n, **attr)

    @property
    def t_ns(self) -> str:
        """Type-level namespace fragment of the underlying recipe."""
        return self.recipe_graph.t_ns

    def get_type(self, node: ANode) -> str:
        return self.recipe_graph.get_type(self.recipe_of(node))  # type: ignore[arg-type]

    @cached_property
    def a_ns(self) -> str:
        """Assertion-level namespace fragment for this graph."""
        h = _get_graph_hash(self, with_global_inputs=True)
        return h + "_"

    @cached_property
    def a_ns_short(self) -> str:
        """Short assertion-level namespace fragment for this graph for RDFS.label."""
        return self.a_ns[:6] + "_"

    def _get_data_node(self, io: IO) -> str:
        while True:
            candidate = [c for c in self.predecessors(io) if isinstance(c, IO)]
            assert len(candidate) <= 1
            if len(candidate) == 0:
                return f"{io}_data"
            io = candidate[0]

    def append_hash(
        self,
        node: str,
        hash_value: str,
        label: str | None = None,
    ):
        """
        Propagates a hash value through the descendants of a given node in a
        directed graph.

        This function iteratively traverses the descendants of the graph and
        appends a hash value to each descendant node. The hash value is
        updated based on the label of each node. Optionally, it can remove
        specific data (e.g., "value") from the nodes.

        Parameters:
            node (str): The starting node from which the hash propagation begins.
            hash_value (str): The initial hash value to propagate through the
                descendants.
            label (str | None, optional): A label to use for hash computation.
                If not provided, the label is derived from the node's data
                (e.g., "label"). Defaults to None.

        Notes:
            - The function uses an iterative approach to avoid recursion,
                making it suitable for graphs with deep hierarchies.
            - The hash value for each node is updated in the format:
                `parent_hash@child_label`.
        """
        stack = [(node, hash_value, label)]

        while stack:
            current_node, current_hash, current_label = stack.pop()

            for child in self.successors(current_node):
                if isinstance(child, Node):
                    continue

                child_label = current_label
                if child_label is None:
                    child_label = self.get_data(child).get("label", child.port)

                self.nodes[child]["hash"] = current_hash + f"@{child_label}"
                stack.append((child, current_hash, child_label))

    def get_hash_dict(self) -> dict[str, str]:
        """
        Get a dictionary mapping node names to their hash values.

        Returns:
            dict[str, str]: A dictionary where keys are node names and values
                are their corresponding hash values.
        """
        hash_dict = {}
        for _, data in self.nodes.data():
            if "hash" in data and "value" in data:
                hash_dict[data["hash"]] = data["value"]
        return hash_dict

    def _remove_constant(self) -> None:
        to_delete = []
        for node in self.nodes:
            if isinstance(node, ANode) and self.get_type(node) == "constant":
                to_delete.extend([node, *self.successors(node)])
        self.remove_nodes_from(to_delete)


def _infer_workflow_label(
    recipe: fr.schemas.WorkflowRecipe, label: str | None = None
) -> str:
    if recipe.reference is None:
        if label is not None:
            return label
        return "workflow"
    return recipe.reference.info.fully_qualified_name.rsplit(".", 1)[-1]


def _output_port_label(port: str, outputs: list[str]) -> str:
    if port == "output_0" and len(outputs) == 1 and outputs[0] == "output_0":
        return "output"
    return port


def _check_for_each_ports(
    nested_ports: list[str],
    zipped_ports: list[str],
    report_unsupported: Callable[[str], Any],
) -> None:
    if len(nested_ports) > 1 or (nested_ports and zipped_ports):
        report_unsupported(
            "Nested ports are only supported as the single iterated port"
        )
    elif len(zipped_ports) == 1:
        report_unsupported("A single iterated port must be given as a nested port")


def _raise_not_implemented(message: str) -> None:
    raise NotImplementedError(message)


DISTRIBUTOR = "distributor"
COLLECTOR = "collector"
FOR_EACH = "for_each"
DISTRIBUTOR_INPUT, DISTRIBUTOR_OUTPUT = "items", "item"
COLLECTOR_INPUT, COLLECTOR_OUTPUT = "item", "items"

_DISTRIBUTOR_FUNCTIONS = (
    transformers.Transform1toN.iterable_to_outputs,
    transformers.Transform1toN.iterable_to_output,
)


def _io_attrs(position: int, port: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {"position": position}
    if type_hint := annotation_to_type_hint(port.annotation):
        attrs["dtype"] = type_hint
    if type_metadata := annotation_to_type_metadata(port.annotation):
        attrs.update(type_metadata.to_dictionary())
    return attrs


def _descends_from(node: ANode | AIO, ancestor: ANode) -> bool:
    current: ANode | None = node.node if isinstance(node, AIO) else node
    while current is not None:
        if current == ancestor:
            return True
        current = current.owner
    return False


class _GraphBuilder:
    """
    Build the recipe graph (T-Box) and the instance graph (A-Box) of a flowrep
    workflow in one pass.

    The T-Box is derived from the recipes only: a for-each node is a single
    class made of a ``distributor`` per iterated input, one body node and a
    ``collector`` per output, irrespective of the number of iterations. The
    A-Box mirrors the executed data: one body instance (``ANode.counter``) per
    iteration, and one distributor output / collector input
    (``AIO.counter``) per scattered / gathered item. A for-each node which has
    not been executed is represented by a single body instance.
    """

    def __init__(self, root_label: str, prefix: str | None):
        self.R = SemantikonRecipeGraph(prefix=prefix)
        self.R.name = prefix if prefix is not None else root_label
        self.A = SemantikonInstanceGraph(recipe_graph=self.R)
        self.constants: dict[IO, Any] = {}

    def link(self, r_edge: tuple[Node | IO, Node | IO], a_edge) -> None:
        self.R.add_edge(*r_edge)
        self.A.add_edge(*a_edge)

    def add_node(
        self,
        node_data: fr.schemas.NodeData,
        r_name: Node,
        a_name: ANode,
        *,
        workflow_label: Node | None = None,
    ) -> None:
        metadata: dict[str, Any] = {}
        function = None
        if isinstance(node_data, fr.schemas.AtomicData):
            function = node_data.function
        elif isinstance(node_data, fr.schemas.ConstantData):
            metadata["type"] = "constant"
        elif isinstance(node_data, fr.schemas.ForEachData):
            metadata["type"] = FOR_EACH
        elif isinstance(node_data, fr.schemas.DagData):
            if workflow_label is not None:
                metadata["label"] = str(workflow_label)
            if node_data.recipe.reference is not None:
                function = retrieve.import_from_string(
                    node_data.recipe.reference.info.fully_qualified_name
                )
        else:
            raise NotImplementedError(
                f"{type(node_data).__name__} is not supported yet."
            )

        if function is not None:
            function_data = get_function_dict(function)
            function_data["identifier"] = ".".join(
                (
                    function_data["module"],
                    function_data["qualname"],
                    function_data["version"],
                )
            )
            metadata["function"] = function_data
        self.R.add_node(r_name, **metadata)
        self.A.add_node(a_name, recipe=r_name)

        output_labels = list(node_data.output_ports)
        if len(output_labels) == 1 and output_labels[0] == "output_0":
            output_labels = ["output"]

        for position, (label, port) in enumerate(node_data.input_ports.items()):
            r_io = Input(node=r_name, port=label)
            io_data = _io_attrs(position, port)
            if not isinstance(port.default, fr.schemas.NotData):
                io_data["default"] = port.default
            self.R.add_node(r_io, **io_data)
            self.R.add_edge(r_io, r_name)
            a_io = AInput(node=a_name, port=label)
            self._add_a_io(a_io, r_io, port)
            self.A.add_edge(a_io, a_name)
        for position, (raw_label, port) in enumerate(node_data.output_ports.items()):
            label = output_labels[position] if raw_label == "output_0" else raw_label
            r_out = Output(node=r_name, port=label)
            self.R.add_node(r_out, **_io_attrs(position, port))
            self.R.add_edge(r_name, r_out)
            a_out = AOutput(node=a_name, port=label)
            self._add_a_io(a_out, r_out, port)
            self.A.add_edge(a_name, a_out)
            if isinstance(node_data, fr.schemas.ConstantData):
                self.constants.setdefault(r_out, port.value)

        if isinstance(node_data, fr.schemas.DagData):
            self._add_dag(node_data, r_name, a_name)
        elif isinstance(node_data, fr.schemas.ForEachData):
            self._add_for_each(node_data, r_name, a_name)

    def _add_a_io(self, a_io: AIO, r_io: IO, port: Any = None) -> None:
        attrs: dict[str, Any] = {"recipe": r_io}
        if port is not None and not isinstance(port.value, fr.schemas.NotData):
            attrs["value"] = port.value
        self.A.add_node(a_io, **attrs)

    def _add_dag(
        self, node_data: fr.schemas.DagData, r_name: Node, a_name: ANode
    ) -> None:
        recipe = node_data.recipe
        for child_label, child in node_data.nodes.items():
            self.add_node(
                child,
                Node(name=child_label, owner=r_name),
                ANode(name=child_label, owner=a_name),
                workflow_label=(
                    Node(child_label) if isinstance(child, fr.schemas.DagData) else None
                ),
            )

        def r_child(label: str) -> Node:
            return Node(owner=r_name, name=label)

        def a_child(label: str) -> ANode:
            return ANode(owner=a_name, name=label)

        child_recipes = recipe.nodes
        for target, source in recipe.input_edges.items():
            self.link(
                (
                    Input(node=r_name, port=source.port),
                    Input(node=r_child(target.node), port=target.port),
                ),
                (
                    AInput(node=a_name, port=source.port),
                    AInput(node=a_child(target.node), port=target.port),
                ),
            )
        for target, source in recipe.edges.items():
            src_port = _output_port_label(
                source.port, list(child_recipes[source.node].outputs)
            )
            self.link(
                (
                    Output(node=r_child(source.node), port=src_port),
                    Input(node=r_child(target.node), port=target.port),
                ),
                (
                    AOutput(node=a_child(source.node), port=src_port),
                    AInput(node=a_child(target.node), port=target.port),
                ),
            )
        for target, source in recipe.output_edges.items():
            target_port = _output_port_label(target.port, list(recipe.outputs))
            if isinstance(source, fr.schemas.InputSource):
                self.link(
                    (
                        Input(node=r_name, port=source.port),
                        Output(node=r_name, port=target_port),
                    ),
                    (
                        AInput(node=a_name, port=source.port),
                        AOutput(node=a_name, port=target_port),
                    ),
                )
            else:
                src_port = _output_port_label(
                    source.port, list(child_recipes[source.node].outputs)
                )
                self.link(
                    (
                        Output(node=r_child(source.node), port=src_port),
                        Output(node=r_name, port=target_port),
                    ),
                    (
                        AOutput(node=a_child(source.node), port=src_port),
                        AOutput(node=a_name, port=target_port),
                    ),
                )

    def _add_for_each(
        self, node_data: fr.schemas.ForEachData, r_name: Node, a_name: ANode
    ) -> None:
        recipe = node_data.recipe
        body_label = recipe.body_node.label
        # A single iterated port is read back as nested, several as zipped
        _check_for_each_ports(
            recipe.nested_ports, recipe.zipped_ports, _raise_not_implemented
        )
        iterated = set(recipe.iterated_ports)
        distributor_sources = {
            target.port: source.port
            for target, source in recipe.input_edges.items()
            if target.port in iterated
        }
        source_counts: dict[str, int] = {}
        for parent_port in distributor_sources.values():
            source_counts[parent_port] = source_counts.get(parent_port, 0) + 1
        distributors = {
            body_port: Node(
                name=(
                    f"{DISTRIBUTOR}_{parent_port}"
                    if source_counts[parent_port] == 1
                    else f"{DISTRIBUTOR}_{parent_port}_{body_port}"
                ),
                owner=r_name,
            )
            for body_port, parent_port in distributor_sources.items()
        }
        collectors = {
            port: Node(name=f"{COLLECTOR}_{port}", owner=r_name)
            for port in recipe.outputs
        }
        r_body = Node(name=body_label, owner=r_name)
        generated = [n.name for n in (*distributors.values(), *collectors.values())]
        if body_label in generated:
            raise ValueError(
                f"The body label {body_label!r} collides with a generated node name."
            )

        for port, node in distributors.items():
            self._add_transfer_node(
                node,
                DISTRIBUTOR,
                DISTRIBUTOR_INPUT,
                DISTRIBUTOR_OUTPUT,
                position=(
                    recipe.zipped_ports.index(port)
                    if port in recipe.zipped_ports
                    else 0
                ),
            )
        for node in collectors.values():
            self._add_transfer_node(node, COLLECTOR, COLLECTOR_INPUT, COLLECTOR_OUTPUT)

        executed = bool(node_data.nodes)
        body_labels = [
            label
            for label, child in node_data.nodes.items()
            if not _is_transfer_data(child)
        ]
        # The class is defined by the recipe, so a body is needed even if no
        # iteration was spawned (not executed, or an empty iteration)
        body_data = (
            node_data.nodes[body_labels[0]]
            if body_labels
            else fr.retrospective.datastructures.recipe2data(recipe.body_node.recipe)
        )
        if not body_labels:
            a_body_0 = ANode(name=body_label, owner=a_name)
            self._add_body(body_data, r_body, a_body_0)
            if executed:
                self.A.remove_nodes_from(
                    [n for n in self.A.nodes if _descends_from(n, a_body_0)]
                )
        body_outputs = list(body_data.output_ports)

        def dist_out(body_port: str) -> Output:
            return Output(node=distributors[body_port], port=DISTRIBUTOR_OUTPUT)

        def dist_out_for_parent(parent_port: str) -> Output:
            body_port = next(
                port
                for port, source_port in distributor_sources.items()
                if source_port == parent_port
            )
            return dist_out(body_port)

        for port, node in distributors.items():
            self.R.add_edge(
                Input(node=r_name, port=distributor_sources[port]),
                Input(node=node, port=DISTRIBUTOR_INPUT),
            )
        for target, source in recipe.input_edges.items():
            if target.port in iterated:
                src: IO = dist_out(target.port)
            else:
                src = Input(node=r_name, port=source.port)
            self.R.add_edge(src, Input(node=r_body, port=target.port))
        for target, source in recipe.output_edges.items():
            collector = collectors[target.port]
            if isinstance(source, fr.schemas.InputSource):
                src = dist_out_for_parent(source.port)
            else:
                src = Output(
                    node=r_body, port=_output_port_label(source.port, body_outputs)
                )
            self.R.add_edge(src, Input(node=collector, port=COLLECTOR_INPUT))
            self.R.add_edge(
                Output(node=collector, port=COLLECTOR_OUTPUT),
                Output(
                    node=r_name,
                    port=_output_port_label(target.port, list(recipe.outputs)),
                ),
            )

        if executed:
            self._add_executed_for_each_instances(
                node_data,
                r_name,
                a_name,
                distributors,
                distributor_sources,
                collectors,
                body_labels,
            )
        else:
            self._add_single_for_each_instance(
                recipe,
                r_name,
                a_name,
                distributors,
                distributor_sources,
                collectors,
                body_outputs,
            )

    def _add_body(
        self, body_data: fr.schemas.NodeData, r_body: Node, a_body: ANode
    ) -> None:
        self.add_node(
            body_data,
            r_body,
            a_body,
            workflow_label=(
                Node(r_body.name) if isinstance(body_data, fr.schemas.DagData) else None
            ),
        )

    def _add_transfer_node(
        self,
        r_node: Node,
        node_type: str,
        input_port: str,
        output_port: str,
        position: int = 0,
        **attrs: Any,
    ) -> None:
        self.R.add_node(r_node, type=node_type, **attrs)
        r_in = Input(node=r_node, port=input_port)
        r_out = Output(node=r_node, port=output_port)
        self.R.add_node(r_in, position=position)
        self.R.add_node(r_out, position=0)
        self.R.add_edge(r_in, r_node)
        self.R.add_edge(r_node, r_out)

    def _add_a_transfer_node(
        self,
        r_node: Node,
        a_node: ANode,
        data: fr.schemas.NodeData | None = None,
    ) -> None:
        """
        Instantiate a distributor / collector. Each item is one port counter; without
        data (not executed) there is exactly one item on either side.
        """
        self.A.add_node(a_node, recipe=r_node)
        input_port, output_port = {
            DISTRIBUTOR: (DISTRIBUTOR_INPUT, DISTRIBUTOR_OUTPUT),
            COLLECTOR: (COLLECTOR_INPUT, COLLECTOR_OUTPUT),
        }[self.R.nodes[r_node]["type"]]
        inputs = list(data.input_ports.values()) if data else [None]
        outputs = list(data.output_ports.values()) if data else [None]
        for counter, in_port in enumerate(inputs):
            a_in = AInput(node=a_node, port=input_port, counter=counter)
            self._add_a_io(a_in, Input(node=r_node, port=input_port), in_port)
            self.A.add_edge(a_in, a_node)
        for counter, out_port in enumerate(outputs):
            a_out = AOutput(node=a_node, port=output_port, counter=counter)
            self._add_a_io(a_out, Output(node=r_node, port=output_port), out_port)
            self.A.add_edge(a_node, a_out)

    def _add_single_for_each_instance(
        self,
        recipe: fr.schemas.ForEachRecipe,
        r_name: Node,
        a_name: ANode,
        distributors: dict[str, Node],
        distributor_sources: dict[str, str],
        collectors: dict[str, Node],
        body_outputs: list[str],
    ) -> None:
        """A-Box of a for-each node that has not been executed: the R edges, lifted."""
        a_body = ANode(name=recipe.body_node.label, owner=a_name)
        a_distributors = {
            port: ANode(name=node.name, owner=a_name)
            for port, node in distributors.items()
        }
        a_collectors = {
            port: ANode(name=node.name, owner=a_name)
            for port, node in collectors.items()
        }
        for port, node in distributors.items():
            self._add_a_transfer_node(node, a_distributors[port])
            self.A.add_edge(
                AInput(node=a_name, port=distributor_sources[port]),
                AInput(node=a_distributors[port], port=DISTRIBUTOR_INPUT),
            )
        for port, node in collectors.items():
            self._add_a_transfer_node(node, a_collectors[port])

        iterated = set(recipe.iterated_ports)

        def a_dist_out(body_port: str) -> AOutput:
            return AOutput(node=a_distributors[body_port], port=DISTRIBUTOR_OUTPUT)

        def a_dist_out_for_parent(parent_port: str) -> AOutput:
            body_port = next(
                port
                for port, source_port in distributor_sources.items()
                if source_port == parent_port
            )
            return a_dist_out(body_port)

        for target, source in recipe.input_edges.items():
            a_src: AIO = (
                a_dist_out(target.port)
                if target.port in iterated
                else AInput(node=a_name, port=source.port)
            )
            self.A.add_edge(a_src, AInput(node=a_body, port=target.port))
        for target, source in recipe.output_edges.items():
            if isinstance(source, fr.schemas.InputSource):
                a_src = a_dist_out_for_parent(source.port)
            else:
                a_src = AOutput(
                    node=a_body, port=_output_port_label(source.port, body_outputs)
                )
            a_collector = a_collectors[target.port]
            self.A.add_edge(a_src, AInput(node=a_collector, port=COLLECTOR_INPUT))
            self.A.add_edge(
                AOutput(node=a_collector, port=COLLECTOR_OUTPUT),
                AOutput(
                    node=a_name,
                    port=_output_port_label(target.port, list(recipe.outputs)),
                ),
            )

    def _add_executed_for_each_instances(
        self,
        node_data: fr.schemas.ForEachData,
        r_name: Node,
        a_name: ANode,
        distributors: dict[str, Node],
        distributor_sources: dict[str, str],
        collectors: dict[str, Node],
        body_labels: list[str],
    ) -> None:
        """
        A-Box of an executed for-each node, following the recorded edges. Each
        recorded child is resolved to an A-Box node and a per-port counter.
        """
        recipe = node_data.recipe
        body_label = recipe.body_node.label
        distributor_for_parent = {
            parent_port: body_port
            for body_port, parent_port in distributor_sources.items()
        }
        a_inputs: dict[str, Callable[[str], AIO]] = {}
        a_outputs: dict[str, Callable[[str], AIO]] = {}

        for counter, label in enumerate(body_labels):
            child = node_data.nodes[label]
            a_body = ANode(name=body_label, owner=a_name, counter=counter)
            self._add_body(child, Node(name=body_label, owner=r_name), a_body)
            a_inputs[label] = _a_port_resolver(AInput, a_body)
            a_outputs[label] = _a_port_resolver(
                AOutput, a_body, outputs=list(child.output_ports)
            )

        for label, child in node_data.nodes.items():
            if _is_transfer_data(child, DISTRIBUTOR):
                parent_port = node_data.input_edges[
                    fr.schemas.TargetHandle(
                        node=label, port=transformers.Transform1toN.input_label
                    )
                ].port
                r_node = distributors[distributor_for_parent[parent_port]]
                a_node = ANode(name=r_node.name, owner=a_name)
                self._add_a_transfer_node(r_node, a_node, child)
                a_inputs[label] = _a_port_resolver(
                    AInput, a_node, port=DISTRIBUTOR_INPUT
                )
                a_outputs[label] = _a_port_resolver(
                    AOutput,
                    a_node,
                    port=DISTRIBUTOR_OUTPUT,
                    counters={p: i for i, p in enumerate(child.output_ports)},
                )
            elif _is_transfer_data(child, COLLECTOR):
                output_port = next(
                    target.port
                    for target, source in node_data.output_edges.items()
                    if source.node == label
                )
                r_node = collectors[output_port]
                a_node = ANode(name=r_node.name, owner=a_name)
                self._add_a_transfer_node(r_node, a_node, child)
                a_inputs[label] = _a_port_resolver(
                    AInput,
                    a_node,
                    port=COLLECTOR_INPUT,
                    counters={p: i for i, p in enumerate(child.input_ports)},
                )
                a_outputs[label] = _a_port_resolver(
                    AOutput, a_node, port=COLLECTOR_OUTPUT
                )

        for target, source in node_data.input_edges.items():
            self.A.add_edge(
                AInput(node=a_name, port=source.port),
                a_inputs[target.node](target.port),
            )
        for target, source in node_data.edges.items():
            self.A.add_edge(
                a_outputs[source.node](source.port),
                a_inputs[target.node](target.port),
            )
        for target, source in node_data.output_edges.items():
            self.A.add_edge(
                a_outputs[source.node](source.port),
                AOutput(
                    node=a_name,
                    port=_output_port_label(target.port, list(recipe.outputs)),
                ),
            )


def _a_port_resolver(
    cls: type[AIO],
    node: ANode,
    *,
    port: str | None = None,
    counters: dict[str, int] | None = None,
    outputs: list[str] | None = None,
) -> Callable[[str], AIO]:
    """
    Map a port label recorded by the for-each runtime to the A-Box port: ``port``
    replaces the label, ``counters`` gives the item counter of each label, and
    ``outputs`` (all output labels of the node) applies the single-output renaming.
    """

    def resolve(label: str) -> AIO:
        name = port if port is not None else label
        if port is None and outputs is not None:
            name = _output_port_label(label, outputs)
        return cls(node=node, port=name, counter=counters[label] if counters else 0)

    return resolve


def _is_transfer_data(data: fr.schemas.NodeData, kind: str | None = None) -> bool:
    """Whether the data is a scatter/gather node created by the for-each runtime."""
    if not isinstance(data, fr.schemas.AtomicData):
        return False
    is_distributor = data.function in _DISTRIBUTOR_FUNCTIONS
    is_collector = data.function == transformers.TransformNto1.inputs_to_list
    if kind == DISTRIBUTOR:
        return is_distributor
    if kind == COLLECTOR:
        return is_collector
    return is_distributor or is_collector


def _workflow_to_instance_graph(
    workflow: fr.schemas.DagData,
    *,
    prefix: str | None = None,
) -> tuple[SemantikonInstanceGraph, dict[IO, Any]]:
    """
    Build the instance graph (A-Box) together with its recipe graph (T-Box), and
    the values of the constants of the latter.
    """
    root_label = _infer_workflow_label(workflow.recipe)
    builder = _GraphBuilder(root_label, prefix)
    builder.add_node(
        workflow,
        Node(root_label),
        ANode(root_label),
        workflow_label=Node(root_label),
    )
    return builder.A, builder.constants


def _get_hashed_node_dict_from_graph(
    G: SemantikonInstanceGraph,
) -> dict[str, dict[str, Any]]:
    hash_dict: dict[str, dict[str, Any]] = {}
    for node in nx.topological_sort(G):
        data = G.nodes[node]
        merged = G.get_data(node)
        if isinstance(node, IO):
            for term in ("hash", "value"):
                if term in data:
                    continue
                for predecessor in G.predecessors(node):
                    predecessor_data = G.nodes[predecessor]
                    if term in predecessor_data:
                        data[term] = predecessor_data[term]
                        break
            continue

        hash_dict_tmp: dict[str, Any] = {
            "inputs": {},
            "outputs": [
                G.get_data(out).get("label", out.port) for out in G.successors(node)
            ],
            "node": copy.deepcopy(merged.get("function")),
        }
        if hash_dict_tmp["node"] is None:
            continue
        hash_dict_tmp["node"]["connected_inputs"] = []
        missing_input = False
        for inp in G.predecessors(node):
            inp_data = G.get_data(inp)
            if "hash" in inp_data:
                hash_dict_tmp["inputs"][inp.port] = inp_data["hash"]
                hash_dict_tmp["node"]["connected_inputs"].append(inp.port)
            elif "value" in inp_data or "default" in inp_data:
                value = inp_data.get("value", inp_data.get("default"))
                if is_dataclass(value) and not isinstance(value, type):
                    hash_dict_tmp["inputs"][inp.port] = asdict(value)
                else:
                    hash_dict_tmp["inputs"][inp.port] = value
            else:
                missing_input = True
                break
        if missing_input:
            continue
        h = sha256(
            json.dumps(hash_dict_tmp, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for out in G.successors(node):
            G.nodes[out]["hash"] = h + "@" + G.get_data(out).get("label", out.port)
        hash_dict_tmp["hash"] = h
        hash_dict[node] = hash_dict_tmp
    return hash_dict


def _remove_constant(G: SemantikonRecipeGraph, values: dict[IO, Any]) -> None:
    to_delete = []
    for node in G.nodes:
        if isinstance(node, Node) and G.get_type(node) == "constant":
            output_node = next(iter(G.successors(node)))
            const_value = values[output_node]
            to_delete.extend([node, output_node])
            for inp in G.successors(output_node):
                G.nodes[inp]["constant_value"] = const_value
    G.remove_nodes_from(to_delete)


def _to_dagdata(
    workflow: fr.schemas.DagData | fr.schemas.WorkflowRecipe,
) -> fr.schemas.DagData:
    if isinstance(workflow, fr.schemas.WorkflowRecipe):
        workflow = fr.schemas.DagData.from_recipe(workflow)
    if not isinstance(workflow, fr.schemas.DagData):
        raise TypeError(
            f"Invalid workflow type. Expected flowrep"
            f" {fr.schemas.DagData.__name__!r}, or flowrep"
            f" {fr.schemas.WorkflowRecipe.__name__!r}, but got {type(workflow)}."
        )
    return workflow


def serialize_instance_to_networkx(
    workflow: fr.schemas.DagData | fr.schemas.WorkflowRecipe,
    hash_data: bool = True,
    prefix: str | None = None,
) -> SemantikonInstanceGraph:
    """
    Serialize a flowrep workflow into a SemantikonInstanceGraph (A-Box),
    optionally hashing node data. The corresponding recipe is available as
    ``G.recipe_graph``.

    Args:
        workflow (DagData | WorkflowRecipe): Workflow representation.
        hash_data (bool): Whether to hash node data.
        prefix (str | None): Optional fixed prefix for type-level namespace
            fragments.

    Returns:
        SemantikonInstanceGraph: The serialized workflow instance graph.
    """
    dag = _to_dagdata(workflow)
    A, values = _workflow_to_instance_graph(dag, prefix=prefix)
    R = A.recipe_graph
    if hash_data:
        try:
            hashed_dict = _get_hashed_node_dict_from_graph(A)
        except Exception as e:
            raise RuntimeError(
                "Failed to hash workflow data - use only hashable inputs or set"
                " hash_data=False"
            ) from e
        for node, data in hashed_dict.items():
            A.append_hash(node, data["hash"])
    A._remove_constant()
    _remove_constant(R, values)
    return A


def serialize_and_convert_to_networkx(
    workflow: fr.schemas.DagData | fr.schemas.WorkflowRecipe,
    hash_data: bool = True,
    prefix: str | None = None,
) -> SemantikonRecipeGraph:
    """
    Serialize a flowrep workflow into a SemantikonRecipeGraph (T-Box). It does
    not contain any values; use :func:`serialize_instance_to_networkx` for the
    instance graph.

    Args:
        workflow (DagData | WorkflowRecipe): Workflow representation.
        hash_data (bool): Kept for backward compatibility; hashes belong to
            the instance graph and are not computed here.
        prefix (str | None): Optional fixed prefix for type-level namespace
            fragments.

    Returns:
        SemantikonRecipeGraph: The serialized workflow recipe graph.
    """
    return serialize_instance_to_networkx(
        workflow, hash_data=False, prefix=prefix
    ).recipe_graph


class _HashGraph:
    def _normalize(self, obj: Any) -> Any:
        """
        Convert objects into a deterministic, JSON-safe representation.
        """
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: self._normalize(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._normalize(v) for k, v in obj.items()}
        elif isinstance(obj, IdentifiedNode):
            if isinstance(obj, BNode):
                raise TypeError("Blank nodes cannot be normalized for hashing.")
            return {"__type__": "URIRef", "value": self._normalize(str(obj))}
        elif isinstance(obj, (list, tuple)):
            return [self._normalize(v) for v in obj]
        elif isinstance(obj, str):
            return unicodedata.normalize("NFC", obj)
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        else:
            raise TypeError(
                f"Unsupported type for normalization: {type(obj)} of value {obj}"
            )

    def _canonical_json(self, data: dict) -> str:
        """
        Deterministic JSON serialization.
        """
        return json.dumps(
            self._normalize(data),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _get_graph_hash(
        self,
        G: SemantikonRecipeGraph | SemantikonInstanceGraph,
        with_global_inputs: bool = True,
    ) -> str:
        """
        Generate a deterministic hash for a graph, independent of OS,
        Python version, and non-serializable runtime values.
        """
        G_tmp = nx.DiGraph()

        for node in G.nodes:
            attrs = {}
            data = G.get_data(node)
            if "function" in data:
                func = data["function"]
                attrs["function"] = func.get("hash") or func.get("identifier")
            if G.in_degree(node) == 0 and with_global_inputs:
                if "value" in data:
                    attrs["value"] = data["value"]
                elif "default" in data:
                    attrs["value"] = data["default"]
            if isinstance(node, IO):
                attrs["port"] = node.port
            G_tmp.add_node(node, canon=self._canonical_json(attrs))
        for u, v in G.edges:
            G_tmp.add_edge(u, v)

        return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
            G_tmp,
            node_attr="canon",
        )


def _get_graph_hash(
    G: SemantikonRecipeGraph | SemantikonInstanceGraph,
    with_global_inputs: bool = True,
) -> str:
    """
    Generate a hash for a NetworkX graph, making sure that data types and
    values (except for the global ones) because they can often not be
    serialized.

    Args:
        G (SemantikonRecipeGraph | SemantikonInstanceGraph): input graph
        with_global_inputs (bool): if True, keep values for global inputs

    Returns:
        (str): hash of the graph
    """
    hasher = _HashGraph()
    return hasher._get_graph_hash(G, with_global_inputs)
