#!/bin/bash

tmsmt package://baxter_description/urdf/baxter.urdf allowed-collision.robray tm-blocks.py tm-blocks.pddl regrasp0.robray -q q0.tmp -g regrasp1.robray -o test.out --gui
