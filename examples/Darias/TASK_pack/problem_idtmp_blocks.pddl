;19:18:36 08/11

(define (problem unpack-3blocks)
   (:domain blocks)

   (:objects
          region1__0__0 - location
          region2__0__0 - location
          c1 - block
          c2 - block
          c3 - block
          c4 - block
   )

   (:init
          (handempty)
          (ontable c1 region1__0__0)
          (ontable c2 region1__0__0)
          (ontable c3 region1__0__0)
          (ontable c4 region1__0__0)
          (clear region2__0__0)
          (not (clear region1__0__0))
   )

   (:goal
   (and
        (handempty)
        (ontable c1 region2__0__0)
   ))

)
