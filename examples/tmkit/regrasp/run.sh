#!/bin/bash

for i in {20..23}; do
(
timeout 9m tmsmt package://baxter_description/urdf/baxter.urdf allowed-collision.robray tm-blocks.py tm-blocks.pddl regrasp0.robray -q q0.tmp -g regrasp1.robray -o "output/plan$i.tmp" --gui >> "log/plan$i.log"
)
done
