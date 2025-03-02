# semantikon

[![Push-Pull](https://github.com/pyiron/semantikon/actions/workflows/push-pull.yml/badge.svg)](https://github.com/pyiron/semantikon/actions/workflows/push-pull.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyiron/semantikon/badge.svg?branch=main)](https://coveralls.io/github/pyiron/semantikon?branch=main)

## Overview

In the realm of the workflow management systems, there are well defined inputs and outputs for each node. `semantikon` is a Python package to give scientific context to node inputs and outputs by providing type hinting and interpreters. Therefore, it consists of two **fully** separate parts: type hinting and interpreters.

### **Type hinting**

`semantikon` provides a way to define types for any number of input parameters and any number of output values for function via type hinting, in particular: data type, unit and ontological type. Type hinting is done with the function `u`, which **requires** the type, and **optionally** you can define the units and the ontological type. The type hinting is done in the following way:

```python
>>> from semantikon.typing import u
>>> 
>>> def my_function(
...     distance: u(int, units="meter"),
...     time: u(int, units="second")
... ) -> u(int, units="meter/second", label="speed"):
...     return distance / time

```

`semantikon`'s type hinting does not require to follow any particular standard. It only needs to be compatible with the interpreter applied.

You can also type-hint the inputs and outputs of a function using a class, i.e.:


```python
>>> from semantikon.typing import u
>>> from semantikon.converter import semantikon_class
>>> 
>>> @semantikon_class
... class MyRecord:
...     distance: u(int, units="meter")
...     time: u(int, units="second")
...     result: u(int, units="meter/second", label="speed")
>>> 
>>> def my_function(distance: MyRecord.distance, time: MyRecord.time) -> MyRecord.result:
...     return distance / time

```

This is equivalent to the previous example. Moreover, if you need to modify some parameters, you can use `u` again, e.g. `u(MyRecord.distance, units="kilometer")`.

### **Interpreters**

#### General interpreter

In order to extract argument information, you can use the functions `parse_input_args` and `parse_output_args`. `parse_input_args` parses the input variables and return a dictionary with the variable names as keys and the variable information as values. `parse_output_args` parses the output variables and return a dictionary with the variable information as values if there is one output variable, or a list of dictionaries if it is a tuple.

Example:

```python
>>> from semantikon.typing import u
>>> from semantikon.converter import parse_input_args, parse_output_args
>>> 
>>> def my_function(
...     a: u(int, units="meter"),
...     b: u(int, units="second")
... ) -> u(int, units="meter/second", label="speed"):
...     return a / b
>>> 
>>> print(parse_input_args(my_function))
{'a': {'units': 'meter', 'dtype': <class 'int'>}, 'b': {'units': 'second', 'dtype': <class 'int'>}}

>>> print(parse_output_args(my_function))
{'units': 'meter/second', 'label': 'speed', 'dtype': <class 'int'>}

```

#### Unit conversion with `pint`

`semantikon` provides a way to interpret the types of inputs and outputs of a function via a decorator, in order to check consistency of the types and to convert them if necessary. Currently, `semantikon` provides an interpreter for `pint.UnitRegistry` objects. The interpreter is applied in the following way:

```python
>>> from semantikon.typing import u
>>> from semantikon.converter import units
>>> from pint import UnitRegistry
>>> 
>>> @units
... def my_function(
...     a: u(int, units="meter"),
...     b: u(int, units="second")
... ) -> u(int, units="meter/second", label="speed"):
...     return a / b
>>> 
>>> ureg = UnitRegistry()
>>> 
>>> print(my_function(1 * ureg.meter, 1 * ureg.second))
1.0 meter / second

```

The interpreters check all types and, if necessary, convert them to the expected types **before** the function is executed, in order for all possible errors would be raised before the function execution. The interpreters convert the types in the way that the underlying function would receive the raw values.

In case there are multiple outputs, the type hints are to be passed as a tuple (e.g. `tuple[u(int, "meter"), u(int, "second"))`).

It is not totally garanteed as a functionality, but relative units as given [on this page](https://pint.readthedocs.io/en/0.10.1/wrapping.html#specifying-relations-between-arguments) can be also used.

Interpreters can distinguish between annotated arguments and non-anotated arguments. If the argument is annotated, the interpreter will try to convert the argument to the expected type. If the argument is not annotated, the interpreter will pass the argument as is.

Regardless of type hints are given or not, the interpreter acts only when the input values contain units and ontological types. If the input values do not contain units and ontological types, the interpreter will pass the input values to the function as is.


