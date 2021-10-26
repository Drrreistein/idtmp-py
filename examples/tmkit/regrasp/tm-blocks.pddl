; Task-Motion extension of the 4-operator blocks world domain from the
; 2nd International Planning Competition.

;;; Extensions:
;;; ===========
;;; * Object types for BLOCK and LOCATION
;;; * ONTABLE, PICK-UP, and PUT-DOWN take a second argument for the location

(define (domain blocks)
    (:requirements :typing)
  (:types block - object
          direction - object
          location - object)
  (:predicates (on ?x - block ?y - block)
               (ontable ?x - block ?loc - location)
               (clear ?x)
               (handempty)
               (holding ?x - block))
  (:action pick-up
           :parameters (?x - block ?loc - location ?dir - direction)
           :precondition (and (clear ?x)
                              (handempty)
                              (ontable ?x ?loc)
                              (not (clear ?loc))
                              )
           :effect (and (not (handempty))
                        (clear ?loc)
                        (holding ?x)
                        (not (ontable ?x ?loc))
                        ))
  (:action put-down
           :parameters (?x - block ?loc - location)
           :precondition (and (holding ?x)
                              (clear ?loc))
           :effect (and (not (holding ?x))
                        (handempty)
                        (clear ?x)
                        (ontable ?x ?loc)
                        (not (clear ?loc)))))
