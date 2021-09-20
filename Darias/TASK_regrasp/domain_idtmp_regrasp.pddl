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
                direction - object
                location - object
        )
        (:predicates
                (ontable ?x - block ?loc - location)
                (handempty)
                (holding ?x - block)
                (grasp_dir ?dir - direction)
        )
        (:action pick-up
                :parameters (?x - block ?loc - location ?dir - direction)
                :precondition (and
                        (handempty)
                        (ontable ?x ?loc)
                )
                :effect (and (not (handempty))
                        (holding ?x)
                        (grasp_dir ?dir)
                        (not (ontable ?x ?loc))
                )
        )
        (:action put-down
                :parameters (?x - block ?loc - location)
                :precondition (and (holding ?x)
                        (not (handempty))
                )
                :effect (and (not (holding ?x))
                        (handempty)
                        (ontable ?x ?loc)
                )
        )
)