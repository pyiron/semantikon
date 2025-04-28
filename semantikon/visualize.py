import io
import pydotplus
from IPython.display import Image
from rdflib.tools.rdf2dot import rdf2dot


def visualize(graph):
    """
    Visualize an RDF graph using pydotplus and return a PNG image.

    Args:
        graph (rdflib.Graph): The RDF graph to visualize.

    Returns:
        IPython.display.Image: The PNG image of the RDF graph.
    """
    stream = io.StringIO()
    rdf2dot(graph, stream, opts={"rankdir=LR"})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    return Image(png)
