#!/bin/bash

for i in {25..30}; do
(
timeout 4m tmsmt package://baxter_description/urdf/baxter.urdf allowed-collision.robray tm-blocks.py tm-blocks.pddl 3body-0.robray -q q0.tmp -g 3body-1.robray -o "output/plan'$i'.tmp" --gui >> "log/plan$i.log"
)
done