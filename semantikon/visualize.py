import io
import pydotplus
from IPython.display import Image
from rdflib.tools.rdf2dot import rdf2dot


def visualize(graph):
    stream = io.StringIO()
    rdf2dot(graph, stream, opts={"rankdir=LR"})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    return Image(png)
