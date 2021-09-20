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
                (holding ?x - block)
                (cleaned ?x - block)
                (cooked ?x - block)
                (issink ?x - block)
                (isstove ?x - block)
        )

        (:action pick-up
                :parameters (?x - block ?loc - location)
                :precondition (and
                        (ontable ?x ?loc)
                        (handempty))
                :effect (and
                        (not (ontable ?x ?loc))
                        (not (handempty))
                        (holding ?x))
        )

        (:action put-down
                :parameters (?x - block ?loc - location)
                :precondition (and
                        (holding ?x)
                        (not (handempty)))
                :effect (and
                        (not (holding ?x))
                        (handempty)
                        (ontable ?x ?loc))
        )

        (:action clean
                :parameters (?x - block ?loc - location)
                :precondition (and
                        (issink ?loc)
                        (ontable ?x ?loc)
                        (not (cleaned ?x))
                )
                :effect (and
                        (cleaned ?x)
                        
                )
        )

        (:action cook
                :parameters (?x - block ?loc - location)
                :precondition (and
                        (isstove ?loc)
                        (ontable ?x ?loc)
                        (cleaned ?x)
                        (not (cooked ?x))
                )
                :effect (and
                        (cooked ?x)
                )
        )
)