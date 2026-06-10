import functools
from typing import Any, Callable

from flowrep.api import schemas as frs
from flowrep.api import tools as frt

from semantikon import flowrep_dict


def workflow(func: Callable) -> Callable:
    func = frt.workflow(func)
    # Expose new dictionary getter
    func.get_semantikon_dict = functools.partial(  # type: ignore[attr-defined]
        _get_semantikon_dict, func
    )
    # Override flowrep bound run method
    func.run = functools.partial(run_workflow_dict, func)  # type: ignore[attr-defined]
    return func


def _get_semantikon_dict(workflow_func):
    # Assumes *workflow_func* is already a flowrep workflow recipe holder
    return flowrep_dict.nodedata2dict(
        frs.DagData.from_recipe(workflow_func.flowrep_recipe)
    )


def run_workflow_dict(func, **kwargs) -> dict[str, Any]:
    executed = frt.run_recipe(func.flowrep_recipe, **kwargs)
    return flowrep_dict.nodedata2dict(executed)
