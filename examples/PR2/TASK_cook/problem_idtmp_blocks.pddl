;19:29:02 07/10

(define (problem clean-and-cook)
   (:domain blocks)

   (:objects
          sink__0__0 - location
          stove__0__0 - location
          table__0__0 - location
          box1 - block
          box2 - block
          box3 - block
          box4 - block
          box5 - block
          box6 - block
          box7 - block
          box8 - block
          box9 - block
   )

   (:init
          (handempty)
          (ontable box1 table__0__0)
          (ontable box2 table__0__0)
          (ontable box3 table__0__0)
          (ontable box4 table__0__0)
          (ontable box5 table__0__0)
          (ontable box6 table__0__0)
          (ontable box7 table__0__0)
          (ontable box8 table__0__0)
          (ontable box9 table__0__0)
          (clear sink__0__0)
          (not (clear table__0__0))
          (clear stove__0__0)
          (not (holding box1))
          (not (holding box2))
          (not (holding box3))
          (not (holding box4))
          (not (holding box5))
          (not (holding box6))
          (not (holding box7))
          (not (holding box8))
          (not (holding box9))
          (not (cleaned box1))
          (not (cleaned box2))
          (not (cleaned box3))
          (not (cleaned box4))
          (not (cleaned box5))
          (not (cleaned box6))
          (not (cleaned box7))
          (not (cleaned box8))
          (not (cleaned box9))
          (not (cooked box1))
          (not (cooked box2))
          (not (cooked box3))
          (not (cooked box4))
          (not (cooked box5))
          (not (cooked box6))
          (not (cooked box7))
          (not (cooked box8))
          (not (cooked box9))
          (issink sink__0__0)
          (isstove stove__0__0)
   )

   (:goal
   (and
        (handempty)
        (cooked box1)
        (cooked box2)
   ))

)
