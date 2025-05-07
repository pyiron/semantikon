# semantikon

[![Push-Pull](https://github.com/pyiron/semantikon/actions/workflows/push-pull.yml/badge.svg)](https://github.com/pyiron/semantikon/actions/workflows/push-pull.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyiron/semantikon/badge.svg?branch=main)](https://coveralls.io/github/pyiron/semantikon?branch=main)

<img src="../images/logo.jpeg" alt="Logo" width="300"/>


## Motivation

Let's take a look at the following function:

```python
def get_speed(distance: float, time: float) -> float:
    return distance / time
```

For you as a human, it is clear that this is a function to calculate the speed for a given distance and a time. But for a computer, it is just a function that takes two floats and returns a float. The computer does not know what the inputs and outputs mean. This is where `semantikon` comes in. It provides a way to give scientific context to the inputs and outputs, as well as to the function itself.


## Installation

You can install `semantikon` via `pip`:

```bash
pip install semantikon
```

You can also install `semantikon` via `conda`:

```bash
conda install -c conda-forge semantikon
```

## Overview

In the realm of the workflow management systems, there are well defined inputs and outputs for each node. `semantikon` is a Python package to give scientific context to node inputs and outputs by providing type hinting and interpreters. Therefore, it consists of two **fully** separate parts: type hinting and interpreters.

### **Type hinting**

`semantikon` provides a way to define types for any number of input parameters and any number of output values for function via type hinting, in particular: data type, unit and ontological type. Type hinting is done with the function `u`, which **requires** the type, and **optionally** you can define the units and the ontological type. The type hinting is done in the following way:

```python
>>> from semantikon.typing import u
>>> from rdflib import Namespace
>>>
>>> EX = Namespace("http://example.org/")
>>> 
>>> def get_speed(
...     distance: u(float, units="meter", uri=EX.distance),
...     time: u(float, units="second", uri=EX.time),
... ) -> u(float, units="meter/second", label="speed", uri=EX.speed):
...     return distance / time

```

`semantikon`'s type hinting does not require to follow any particular standard. It only needs to be compatible with the interpreter applied.

You can also type-hint the inputs and outputs of a function using a class, i.e.:


```python
>>> from semantikon.typing import u
>>> from semantikon.converter import semantikon_class
>>> from rdflib import Namespace
>>>
>>> EX = Namespace("http://example.org/")
>>> 
>>> @semantikon_class
... class MyRecord:
...     distance: u(float, units="meter", uri=EX.distance)
...     time: u(float, units="second", uri=EX.time)
...     result: u(float, units="meter/second", label="speed", uri=EX.speed)
>>> 
>>> def get_speed(distance: MyRecord.distance, time: MyRecord.time) -> MyRecord.result:
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
>>> from rdflib import Namespace
>>>
>>> EX = Namespace("http://example.org/")
>>> 
>>> def get_speed(
...     a: u(float, units="meter", uri=EX.distance),
...     b: u(float, units="second", uri=EX.time),
... ) -> u(float, units="meter/second", label="speed", uri=EX.speed):
...     return a / b
>>> 
>>> print(parse_input_args(get_speed))
{'a': {'units': 'meter', 'dtype': <class 'float'>, 'uri': rdflib.term.URIRef('http://example.org/distance')}, 'b': {'units': 'second', 'dtype': <class 'float'>, 'uri': rdflib.term.URIRef('http://example.org/time')}}

>>> print(parse_output_args(get_speed))
{'units': 'meter/second', 'label': 'speed', 'dtype': <class 'float'>, 'uri': rdflib.term.URIRef('http://example.org/speed')}

```

#### Unit conversion with `pint`

`semantikon` provides a way to interpret the types of inputs and outputs of a function via a decorator, in order to check consistency of the types and to convert them if necessary. Currently, `semantikon` provides an interpreter for `pint.UnitRegistry` objects. The interpreter is applied in the following way:

```python
>>> from semantikon.typing import u
>>> from semantikon.converter import units
>>> from pint import UnitRegistry
>>> 
>>> @units
... def get_speed(
...     a: u(float, units="meter"),
...     b: u(float, units="second")
... ) -> u(float, units="meter/second", label="speed"):
...     return a / b
>>> 
>>> ureg = UnitRegistry()
>>> 
>>> print(get_speed(1 * ureg.meter, 1 * ureg.second))
1.0 meter / second

```

The interpreters check all types and, if necessary, convert them to the expected types **before** the function is executed, in order for all possible errors would be raised before the function execution. The interpreters convert the types in the way that the underlying function would receive the raw values.

In case there are multiple outputs, the type hints are to be passed as a tuple (e.g. `tuple[u(float, "meter"), u(float, "second"))`).

It is not totally garanteed as a functionality, but relative units as given [on this page](https://pint.readthedocs.io/en/0.10.1/wrapping.html#specifying-relations-between-arguments) can be also used.

Interpreters can distinguish between annotated arguments and non-anotated arguments. If the argument is annotated, the interpreter will try to convert the argument to the expected type. If the argument is not annotated, the interpreter will pass the argument as is.

Regardless of type hints are given or not, the interpreter acts only when the input values contain units and ontological types. If the input values do not contain units and ontological types, the interpreter will pass the input values to the function as is.


## License

This project is licensed under the BSD 3-Clause License - see the [../LICENSE](LICENSE) file for details.

Copyright (c) 2024, Max-Planck-Institut für Nachhaltige Materialien GmbH - Computational Materials Design (CM) Department
