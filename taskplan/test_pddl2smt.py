#####
# @file

############################################################################
##    This file is part of OMTPlan.
##
##    OMTPlan is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    OMTPlan is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with OMTPlan.  If not, see <https://www.gnu.org/licenses/>.
############################################################################

import sys
import subprocess
import arguments
import translate, utils
from planner import encoder
from planner import modifier
from planner import search
from IPython import embed

def main():
    """
    Main planning routine
    """
    args = arguments.parse_args()

    # Run PDDL translator (from TFD)
    domain = args.domain
    prb = args.problem  

    task = translate.pddl.open_file(prb, domain) 
    # Fetch upper bound for bounded search
    
    ub = 100
    # Compose encoder and search
    # according to user flags
    linear = True
    if linear:
        e = encoder.EncoderSMT(task, modifier.LinearModifier())
        # # Ramp-up search for optimal planning with unit costs
        s = search.SearchSMT(e,ub)
        plan, horizon = s.do_linear_search()
        print('plan found: {} in horizon: {}'.format(plan.plan, horizon))

        # Build SMT-LIB encoding and dump (no solving)
        formula = e.encode(horizon)

        # Print SMT planning formula (linear) to file
        utils.printSMTFormula(formula,task.task_name)
        
    embed()
    
if __name__ == '__main__':
    main()
