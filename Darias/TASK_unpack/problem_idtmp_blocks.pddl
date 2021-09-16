;15:35:48 16/09

(define (problem unpack-3blocks)
   (:domain blocks)

   (:objects
          region1__-1__-1 - location
          region1__-1__0 - location
          region1__-1__1 - location
          region1__0__-1 - location
          region1__0__0 - location
          region1__0__1 - location
          region1__1__-1 - location
          region1__1__0 - location
          region1__1__1 - location
          region2__-1__-1 - location
          region2__-1__0 - location
          region2__-1__1 - location
          region2__0__-1 - location
          region2__0__0 - location
          region2__0__1 - location
          region2__1__-1 - location
          region2__1__0 - location
          region2__1__1 - location
          c1 - block
          c2 - block
          c3 - block
   )

   (:init
          (handempty)
          (ontable c1 region1__0__0)
          (ontable c2 region1__0__0)
          (ontable c3 region1__0__0)
   )

   (:goal
   (and
        (handempty)
        (ontable c1 region2__0__0)
   ))

)
