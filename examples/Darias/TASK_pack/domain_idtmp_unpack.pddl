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
        )

        (:action pick-up
                :parameters (?x - block ?loc - location)
                :precondition (and
                        (ontable ?x ?loc)
                        ; (not (clear ?loc))
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
                        ; (clear ?loc)
                        (not (handempty)))
                :effect (and
                        (not (holding ?x))
                        (handempty)
                        (not (clear ?loc))
                        (ontable ?x ?loc))

        )

)