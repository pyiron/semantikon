import unittest

import flowrep as fr
import semantikon
from semantikon.ontology import SNS


@fr.workflow
def wf(x):
    doubled = fr.std.mul(2, x)
    return doubled


class TestConstants(unittest.TestCase):

    def test_roundtrip(self):
        recipe = wf.flowrep_recipe

        # Probe representation in the knowledge graph
        kg = semantikon.get_knowledge_graph(wf.flowrep_recipe)

        with self.subTest(msg="Verify in A-box"):
            abox_constants = kg.query(f"""
              SELECT ?port ?value WHERE {{
                  ?port <{SNS.has_participant}> ?data .
                  ?data <{SNS.has_value}> ?value .
              }}
            """)
            self.assertEqual(
                len(abox_constants),
                2,
                "The constant should appear as output of one node and input of "
                "the other",
            )

            # We don't care about what order the two query results come, so use a sets
            ports_holding_constant = set()
            for query_result in abox_constants:
                port, literal = query_result
                discovered_port = port.rsplit(f"{wf.__name__}-")[1]
                ports_holding_constant.add(discovered_port)

            self.assertSetEqual(
                {"constant_0-outputs-constant", "mul_0-inputs-a"},
                ports_holding_constant,
                msg="The constant should appear as output of one node and input of the "
                "other",
            )

        with self.subTest(msg="Verify in T-box"):
            tbox_constants = kg.query(f"""
              SELECT ?node ?value WHERE {{
                  ?node rdfs:subClassOf ?r .
                  ?r a owl:Restriction ;
                     owl:onProperty <{SNS.has_value}> ;
                     owl:hasValue ?value .
              }}
            """)
            self.assertEqual(len(tbox_constants), 1)

            node, literal = next(iter(tbox_constants))
            node_label = node.rsplit(f"{wf.__name__}-")[1]
            self.assertEqual(
                "constant_0",
                node_label,
                msg="The constant should be attached to the constant node, like "
                "function references are to atomic nodes",
            )

        with self.subTest(msg="Visualization should not crash"):
            semantikon.visualize_recipe(kg)
            # We don't inspect it here, but it ought at least not crash

        with self.subTest(msg="Verify recipe roundtrip"):
            roundtrip_recipe = semantikon.kg2recipe(kg)
            self.assertEqual(
                recipe,
                roundtrip_recipe,
                msg="The KG should recompile into the same recipe",
            )
            self.assertEqual(6, recipe(x=3), msg="sanity check")
            self.assertEqual(
                recipe(x=3),
                roundtrip_recipe(x=3),
                msg="The re-constituted recipe should execute to the same value",
            )
