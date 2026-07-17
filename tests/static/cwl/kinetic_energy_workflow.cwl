#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow

inputs:
  distance:
    type: float
  time:
    type: float
  mass:
    type: float

outputs:
  kinetic_energy:
    type: float
    outputSource: get_kinetic_energy/kinetic_energy

steps:
  get_speed:
    run: get_speed.cwl
    in:
      distance: distance
      time: time
    out:
      - speed

  get_kinetic_energy:
    run: get_kinetic_energy.cwl
    in:
      mass: mass
      velocity: get_speed/speed
    out:
      - kinetic_energy
