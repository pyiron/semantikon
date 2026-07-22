import functools
import warnings
from typing import Any, Callable

import flowrep as fr

from semantikon import flowrep_dict


def workflow(func: Callable) -> Callable:
    warnings.warn(
        "semantikon workflow decorator is deprecated, use flowrep.workflow instead",
        DeprecationWarning,
    )
    func = fr.tools.workflow(func)
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
        fr.schemas.DagData.from_recipe(workflow_func.flowrep_recipe)
    )


def run_workflow_dict(func, **kwargs) -> dict[str, Any]:
    warnings.warn(
        "semantikon workflow run method is deprecated, use"
        " flowrep.wfms.run_recipe instead",
        DeprecationWarning,
    )
    executed = fr.tools.run_recipe(func.flowrep_recipe, **kwargs)
    return flowrep_dict.nodedata2dict(executed)
