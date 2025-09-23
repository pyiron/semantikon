from collections import defaultdict

import numpy as np
import pandas as pd
import unittest
from ase import Atoms, build
from pint import UnitRegistry
from rdflib import OWL, RDFS, Graph, Namespace
from structuretoolkit import get_neighbors

from semantikon.metadata import u
from semantikon.ontology import get_knowledge_graph
from semantikon.workflow import workflow

EX = Namespace("http://example.org/")


# https://periodictable.com/Properties/A/BulkModulus.al.html
bulk_modulus_dict = {"Al": 76, "Cu": 140, "Ni": 180}


def get_lj_energy(distances, sigma, epsilon):
    r = np.atleast_1d(distances)
    return epsilon * np.sum((sigma / r) ** 12 - (sigma / r) ** 6) / 2


def get_lj_pw_forces(distances, vecs, sigma, epsilon):
    v = vecs.copy()
    v[v == np.inf] = 0
    return (
        -6
        * epsilon
        * ((2 * (sigma / distances) ** 12 - (sigma / distances) ** 6) / distances**2)[
            ..., None
        ]
        * v
    )


def get_lj_forces(distances, vecs, sigma, epsilon):
    return get_lj_pw_forces(distances, vecs, sigma, epsilon).sum(axis=1)


def get_lj_pressure(distances, vecs, sigma, epsilon, volume):
    v = vecs.copy()
    v[v == np.inf] = 0
    p = (
        -0.25
        * np.einsum(
            "...i,...j->...ij", get_lj_pw_forces(distances, v, sigma, epsilon), v
        ).sum(axis=(0, 1))
        / volume
    )
    return p + p.T


def get_sigma(
    structure: Atoms, cutoff=4, sigma_start=2, d_sigma=1, epsilon=1, n_cycle=20
):
    sigma = sigma_start
    p_hist = []
    for _ in range(20):
        neigh = get_neighbors(
            structure, num_neighbors=None, cutoff_radius=cutoff * sigma
        )
        p = get_lj_pressure(
            neigh.distances, neigh.vecs, sigma, epsilon, structure.get_volume()
        )[0, 0]
        p_hist.append([sigma, p])
        if p < 0:
            sigma += d_sigma
        else:
            sigma -= d_sigma
        d_sigma *= 0.5
    p_hist = np.array(p_hist)
    return sigma


def get_bulk_modulus(structure, sigma, epsilon, cutoff=4, factor=0.99):
    neigh = get_neighbors(structure, num_neighbors=None, cutoff_radius=cutoff * sigma)
    p = get_lj_pressure(
        neigh.distances, neigh.vecs, sigma, epsilon, structure.get_volume()
    )
    p_m = get_lj_pressure(
        neigh.distances * factor,
        neigh.vecs * factor,
        sigma,
        epsilon,
        structure.get_volume() * factor**3,
    )
    return -(p - p_m)[0, 0] / (1 - factor**3)


def get_epsilon(
    structure,
    sigma,
    epsilon_start=0.6,
    d_epsilon=0.5,
    n_cycle=20,
    cutoff=4,
    factor=0.99,
    bulk_modulus_dict=bulk_modulus_dict,
):
    ureg = UnitRegistry()
    ref_bulk_modulus = bulk_modulus_dict[structure.get_chemical_symbols()[0]]
    epsilon = epsilon_start
    e_hist = []
    for n in range(n_cycle):
        B = (
            (
                get_bulk_modulus(structure, sigma, epsilon)
                * ureg.electron_volt
                / ureg.angstrom**3
            )
            .to("gigapascal")
            .magnitude
        )
        e_hist.append([epsilon, B])
        if B > ref_bulk_modulus:
            epsilon -= d_epsilon
        else:
            epsilon += d_epsilon
        d_epsilon *= 0.5
    return epsilon


def get_cutoff_radius(sigma, mul=4):
    cutoff_radius = sigma * mul
    return cutoff_radius


def get_BFGS(dx, dg, H):
    Hx = H.dot(dx)
    return np.outer(dg, dg) / dg.dot(dx) - np.outer(Hx, Hx) / dx.dot(Hx) + H


def relax_structure(
    structure: Atoms, sigma, epsilon, cutoff_radius, h_init=100, n_max=100, f_max=0.01
) -> u(Atoms, derived_from="inputs.structure"):
    struct = structure.copy()
    H = h_init * np.eye(np.prod(structure.positions.shape))
    x_lst = [struct.positions.copy()]
    neigh = get_neighbors(struct, num_neighbors=None, cutoff_radius=cutoff_radius)
    f_lst = [get_lj_forces(neigh.distances, neigh.vecs, sigma, epsilon)]
    E_hist = []
    for _ in range(n_max):
        struct.positions += (np.linalg.inv(H) @ f_lst[-1].flatten()).reshape(-1, 3)
        neigh = get_neighbors(struct, num_neighbors=None, cutoff_radius=cutoff_radius)
        x_lst.append(struct.positions.copy())
        f_lst.append(get_lj_forces(neigh.distances, neigh.vecs, sigma, epsilon))
        E_hist.append(
            (
                get_lj_energy(neigh.distances, sigma, epsilon),
                np.linalg.norm(f_lst[-1], axis=-1).max(),
            )
        )
        if E_hist[-1][-1] < f_max:
            break
        dx = np.diff(x_lst, axis=0).reshape(-1, np.prod(structure.positions.shape))
        dg = -np.diff(f_lst, axis=0).reshape(-1, np.prod(structure.positions.shape))
        H = get_BFGS(dx[-1], dg[-1], H)
    return structure


def get_structure(element: str) -> u(Atoms, triples=(EX.hasElement, "inputs.element")):
    structure = build.bulk(element, cubic=True)
    return structure


def repeat_structure(
    structure: Atoms, n_repeat: int
) -> u(Atoms, derived_from="inputs.structure"):
    return structure.repeat(n_repeat)


def create_vacancy(
    structure: Atoms,
) -> u(Atoms, triples=(EX.hasDefect, EX.Vacancy), derived_from="inputs.structure"):
    vac = structure.copy()
    del vac[0]
    return vac


def get_energy(
    structure: Atoms,
    sigma: u(float, units="angstrom"),
    epsilon: u(float, units="electron_volt"),
    cutoff_radius: u(float, units="angstrom"),
    per_atom=False,
) -> u(float, units="eV"):
    neigh = get_neighbors(structure, num_neighbors=None, cutoff_radius=cutoff_radius)
    energy = get_lj_energy(neigh.distances, sigma, epsilon)
    if per_atom:
        energy /= len(structure)
    return energy


def get_energy_difference(
    e_vac, e_bulk: float, structure: Atoms
) -> u(float, derived_from="inputs.e_vac", uri=EX.VacancyFormationEnergy):
    energy = e_vac - e_bulk * len(structure)
    return energy


def get_bulk_energy(
    bulk: Atoms,
    sigma: u(float, units="angstrom"),
    epsilon: u(float, units="electron_volt"),
    cutoff_radius: u(float, units="angstrom"),
) -> u(float, units="eV"):
    return get_energy(
        structure=bulk,
        sigma=sigma,
        epsilon=epsilon,
        cutoff_radius=cutoff_radius,
        per_atom=True,
    )


def get_vac_energy(
    vacancy_structure: u(
        Atoms,
        restrictions=((OWL.onProperty, EX.hasDefect), (OWL.someValuesFrom, EX.Vacancy)),
    ),
    sigma: float,
    epsilon: float,
    cutoff_radius: float,
) -> u(float, triples=(EX.propertyOf, "inputs.vacancy_structure")):
    return get_energy(
        structure=vacancy_structure,
        sigma=sigma,
        epsilon=epsilon,
        cutoff_radius=cutoff_radius,
        per_atom=False,
    )


@workflow
def get_vacancy_formation_energy(element):
    bulk = get_structure(element=element)
    sigma = get_sigma(structure=bulk)
    epsilon = get_epsilon(structure=bulk, sigma=sigma)
    cutoff_radius = get_cutoff_radius(sigma=sigma)
    vacancy = create_vacancy(structure=bulk)
    relaxed_vacancy = relax_structure(
        structure=vacancy, sigma=sigma, epsilon=epsilon, cutoff_radius=cutoff_radius
    )
    bulk_energy = get_bulk_energy(
        bulk=bulk, sigma=sigma, epsilon=epsilon, cutoff_radius=cutoff_radius
    )
    vac_energy = get_vac_energy(
        vacancy_structure=relaxed_vacancy,
        sigma=sigma,
        epsilon=epsilon,
        cutoff_radius=cutoff_radius,
    )
    vacancy_formation_energy = get_energy_difference(
        vac_energy, bulk_energy, relaxed_vacancy
    )

    return vacancy_formation_energy


query = """
PREFIX ex: <http://example.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?element ?energy
WHERE {
    ?output_tag rdf:value ?energy .
    ?output sns:hasValue ?output_tag .
    ?output rdf:type ex:VacancyFormationEnergy .
    ?output ex:propertyOf ?structure .
    ?structure ex:hasElement ?input_element .
    ?input_element sns:hasValue ?input_element_tag .
    ?input_element_tag rdf:value ?element
}"""


class TestVacancyFormationEnergy(unittest.TestCase):
    def test_vacancy_formation_energy(self):
        graph = Graph()
        graph.add((EX.HasDefect, RDF.type, RDF.Property))
        graph.add((EX.Vacancy, RDF.type, OWL.Class))
        for elem in ["Al", "Cu", "Ni"]:
            wf_graph = get_vacancy_formation_energy.run(element=elem)
            graph = get_knowledge_graph(wf_graph, graph=graph, use_uuid=True)

        data = defaultdict(list)

        for row in graph.query(query):
            data["element"].append(row[0].value)
            data["energy"].append(round(row[1].value, 3))

        df = pd.DataFrame(data)
        self.assertEqual(
            df.to_csv(), ",element,energy\n0,Al,0.744\n1,Cu,0.97\n2,Ni,1.157\n"
        )


if __name__ == "__main__":
    unittest.main()
