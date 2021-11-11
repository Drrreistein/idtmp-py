; Task-Motion extension of the 4-operator blocks world domain from the
; 2nd International Planning Competition.

;;; Extensions:
;;; ===========
;;; * Object types for BLOCK and LOCATION
;;; * ONTABLE, PICK-UP, and PUT-DOWN take a second argument for the location

(define (domain blocks)
        (:requirements :typing)
        (:types
                block - object
                location - object
        )

        (:predicates
                (ontable ?x - block ?loc - location)
                (handempty)
                (clear ?loc - location)
                (holding ?x - block)
                (cleaned ?x - block)
                (cooked ?x - block)
                (issink ?x - location)
                (isstove ?x - location)
        )

        (:action move_clean_cook
                :parameters (?x - block ?orig_loc - location ?clean_loc - location ?cook_loc - location)
                :precondition (and
                        (ontable ?x ?orig_loc)
                        (handempty)
                        (not (holding ?x))
                        (not (cooked ?x))
                        (not (cleaned ?x))
                        (issink ?clean_loc)
                        (isstove ?cook_loc)
                )
                :effect (and 
                        (cooked ?x)
                        (handempty)
                )
        )
)
