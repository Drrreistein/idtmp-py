;22:50:20 26/09

(define (problem clean-and-cook)
   (:domain blocks)

   (:objects
          sink__-1__-1 - location
          sink__-1__0 - location
          sink__-1__1 - location
          sink__0__-1 - location
          sink__0__0 - location
          sink__0__1 - location
          sink__1__-1 - location
          sink__1__0 - location
          sink__1__1 - location
          stove__-1__-1 - location
          stove__-1__0 - location
          stove__-1__1 - location
          stove__0__-1 - location
          stove__0__0 - location
          stove__0__1 - location
          stove__1__-1 - location
          stove__1__0 - location
          stove__1__1 - location
          table__-1__-1 - location
          table__-1__0 - location
          table__-1__1 - location
          table__0__-1 - location
          table__0__0 - location
          table__0__1 - location
          table__1__-1 - location
          table__1__0 - location
          table__1__1 - location
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
          (ontable box1 table__-1__-1)
          (ontable box2 table__0__-1)
          (ontable box3 table__1__-1)
          (ontable box4 table__-1__0)
          (ontable box5 table__0__0)
          (ontable box6 table__1__0)
          (ontable box7 table__-1__1)
          (ontable box8 table__0__1)
          (ontable box9 table__1__1)
          (not (clear table__0__0))
          (not (clear table__-1__-1))
          (clear sink__0__-1)
          (clear sink__-1__1)
          (clear stove__1__-1)
          (clear stove__1__1)
          (clear sink__1__-1)
          (clear sink__1__0)
          (clear sink__1__1)
          (clear stove__-1__0)
          (clear stove__-1__-1)
          (not (clear table__1__1))
          (not (clear table__0__-1))
          (clear sink__-1__0)
          (not (clear table__0__1))
          (clear sink__0__1)
          (clear stove__-1__1)
          (clear stove__0__-1)
          (not (clear table__-1__0))
          (clear sink__0__0)
          (clear stove__1__0)
          (clear sink__-1__-1)
          (clear stove__0__1)
          (not (clear table__1__0))
          (clear stove__0__0)
          (not (clear table__-1__1))
          (not (clear table__1__-1))
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
          (issink sink__0__-1)
          (issink sink__-1__1)
          (isstove stove__1__-1)
          (isstove stove__1__1)
          (issink sink__1__-1)
          (issink sink__1__0)
          (issink sink__1__1)
          (isstove stove__-1__0)
          (isstove stove__-1__-1)
          (issink sink__-1__0)
          (issink sink__0__1)
          (isstove stove__-1__1)
          (isstove stove__0__-1)
          (issink sink__0__0)
          (isstove stove__1__0)
          (issink sink__-1__-1)
          (isstove stove__0__1)
          (isstove stove__0__0)
   )

   (:goal
   (and
        (handempty)
        (cooked box1)
        (cooked box2)
   ))

)
