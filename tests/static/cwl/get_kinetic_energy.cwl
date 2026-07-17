#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool

requirements:
  InlineJavascriptRequirement: {}

baseCommand: python3

arguments:
  - prefix: -c
    valueFrom: |
      import sys
      mass = float(sys.argv[1])
      velocity = float(sys.argv[2])
      kinetic_energy = 0.5 * mass * velocity ** 2
      print(kinetic_energy)

inputs:
  mass:
    type: float
    inputBinding:
      position: 1

  velocity:
    type: float
    inputBinding:
      position: 2

outputs:
  kinetic_energy:
    type: float
    outputBinding:
      glob: kinetic_energy_output.txt
      loadContents: true
      outputEval: $(parseFloat(self[0].contents.trim()))

stdout: kinetic_energy_output.txt
