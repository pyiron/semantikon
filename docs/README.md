# semantikon

## Overview

In the realm of the workflow management systems, there are well defined inputs and outputs for each node. `semantikon` is a Python package to give scientific context to node inputs and outputs by providing type hinting and interpreters. Therefore, it consists of two **fully** separate parts: type hinting and interpreters.

### **Type hinting**

`semantikon` provides a way to define types for any number of input parameters and any number of output values for function via type hinting, in particular: data type, unit and ontological type. Type hinting is done with the function `u`, which **requires** the type, and **optionally** you can define the units and the ontological type. The type hinting is done in the following way:

```python
from semantikon.typing import u

def my_function(
    a: u(int, units="meter"),
    b: u(int, units="second")
) -> u(int, units="meter/second", label="speed"):
    return a / b
```

`semantikon`'s type hinting does not require to follow any particular standard. It only needs to be compatible with the interpreter applied.

There are two possible ways to store the data for `semantikon`. The standard way is to do it by converting all arguments except for the data type as a string, which is the default behaviour. The other way is to store the data as a list, which is turned on by setting `use_list=True`. In most cases, the default behaviour is the safest option; in some cases, especially when the data cannot be represented as a string, you might want to switch on `use_list`, but `semantikon` is still under intensive development, and therefore there is no guarantee that you can retrieve the data across different versions correctly.


### **Interpreters**

#### General interpreter

In order to extract argument information, you can use the functions `parse_input_args` and `parse_output_args`. `parse_input_args` parses the input variables and return a dictionary with the variable names as keys and the variable information as values. `parse_output_args` parses the output variables and return a dictionary with the variable information as values if there is one output variable, or a list of dictionaries if it is a tuple.

Example:

```python
from semantikon.typing import u
from semantikon.converter import parse_input_args, parse_output_args

def my_function(
    a: u(int, units="meter"),
    b: u(int, units="second")
) -> u(int, units="meter/second", label="speed"):
    return a / b

print(parse_input_args(my_function))
print(parse_output_args(my_function))
```

Output:

```python
{'distance': {'units': 'meter', 'label': None, 'uri': None, 'shape': None, 'dtype': <class 'float'>}, 'time': {'units': 'second', 'label': None, 'uri': None, 'shape': None, 'dtype': <class 'float'>}}
{'units': 'meter/second', 'label': 'speed', 'uri': None, 'shape': None, 'dtype': <class 'float'>}
```

Here the output is the same whether `use_list` is set to `True` or `False`. When `use_list` is `False`, you can use additionally any tag that you want to store. When `use_list` is `True`, you can store only the data type, `units`, `label`, `uri`, `shape` and `dtype`.

Future announcement: There will be no distrinction between `use_list=True` and `use_list=False` when the official support of python 3.10 is dropped (i.e. around autumn 2026).

#### Unit conversion with `pint`

`semantikon` provides a way to interpret the types of inputs and outputs of a function via a decorator, in order to check consistency of the types and to convert them if necessary. Currently, `semantikon` provides an interpreter for `pint.UnitRegistry` objects. The interpreter is applied in the following way:

```python
from semantikon.typing import u
from semantikon.converters import units
from pint import UnitRegistry

@units
def my_function(
    a: u(int, units="meter"),
    b: u(int, units="second")
) -> u(int, units="meter/second", label="speed"):
    return a / b


ureg = UnitRegistry()

print(my_function(1 * ureg.meter, 1 * ureg.second))
```

Output: `1.0 meter / second`


The interpreters check all types and, if necessary, convert them to the expected types **before** the function is executed, in order for all possible errors would be raised before the function execution. The interpreters convert the types in the way that the underlying function would receive the raw values.

In case there are multiple outputs, the type hints are to be passed as a tuple (e.g. `(u(int, "meter"), u(int, "second"))`).

Interpreters can distinguish between annotated arguments and non-anotated arguments. If the argument is annotated, the interpreter will try to convert the argument to the expected type. If the argument is not annotated, the interpreter will pass the argument as is.

Regardless of type hints are given or not, the interpreter acts only when the input values contain units and ontological types. If the input values do not contain units and ontological types, the interpreter will pass the input values to the function as is.


