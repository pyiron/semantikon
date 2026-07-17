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
      distance = float(sys.argv[1])
      time = float(sys.argv[2])
      speed = distance / time
      print(speed)

inputs:
  distance:
    type: float
    inputBinding:
      position: 1

  time:
    type: float
    inputBinding:
      position: 2

outputs:
  speed:
    type: float
    outputBinding:
      glob: speed_output.txt
      loadContents: true
      outputEval: $(parseFloat(self[0].contents.trim()))

stdout: speed_output.txt
