# uniton

## Overview

In the realm of the workflow management systems, there are well defined inputs and outputs for each node. `uniton` is a Python package to give scientific context to node inputs and outputs by providing type hinting and interpreters. Therefore, it consists of two **fully** separate parts: type hinting and interpreters.

### **Type hinting**

`uniton` provides a way to define types for any number of input parameters and any number of output values for function via type hinting, in particular: data type, unit and ontological type. Type hinting is done with the function `u`, which **requires** the type, and **optionally** you can define the units and the ontological type. The type hinting is done in the following way:

```python
from uniton.typing import u

def my_function(
    a: u(int, "meter", my_ontology_for_length),
    b: u(int, "second", my_ontology_for_time)
) -> u(int, "meter/second", my_ontology_for_speed):
    return a / b
```

`uniton`'s type hinting does not require to follow any particular standard. It only needs to be compatible with the interpreter applied.

### **Interpreters**

`uniton` provides a way to interpret the types of inputs and outputs of a function via a decorator, in order to check consistency of the types and to convert them if necessary. Currently, `uniton` provides an interpreter for `pint.UnitRegistry` objects. The interpreter is applied in the following way:

```python
from uniton.typing import u
from uniton.converters import units
from pint import UnitRegistry

@units
def my_function(
    a: u(int, "meter", my_ontology_for_length),
    b: u(int, "second", my_ontology_for_time)
) -> u(int, "meter/second", my_ontology_for_speed):
    return a / b


ureg = UnitRegistry()

print(my_function(1 * ureg.meter, 1 * ureg.second))
```

Output: `1.0 meter / second`


The interpreters check all types and, if necessary, convert them to the expected types **before** the function is executed, in order for all possible errors would be raised before the function execution. The interpreters convert the types in the way that the underlying function would receive the raw values.

In case there are multiple outputs, the type hints are to be passed as a tuple (e.g. `(u(int, "meter", my_ontology_for_length), u(int, "second", my_ontology_for_time))`).

Interpreters can distinguish between annotated arguments and non-anotated arguments. If the argument is annotated, the interpreter will try to convert the argument to the expected type. If the argument is not annotated, the interpreter will pass the argument as is.

Regardless of type hints are given or not, the interpreter acts only when the input values contain units and ontological types. If the input values do not contain units and ontological types, the interpreter will pass the input values to the function as is.
