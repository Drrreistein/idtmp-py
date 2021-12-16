;15:44:20 16/12

(define (problem unpack-3blocks)
   (:domain blocks)

   (:objects
          region0__-1__-1 - location
          region0__-1__0 - location
          region0__-1__1 - location
          region0__0__-1 - location
          region0__0__0 - location
          region0__0__1 - location
          region0__1__-1 - location
          region0__1__0 - location
          region0__1__1 - location
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
          box0 - block
          box1 - block
          box2 - block
   )

   (:init
          (handempty)
          (ontable box2 region0__0__0)
          (ontable box0 region2__0__0)
          (ontable box1 region0__0__0)
   )

   (:goal
   (and
        (handempty)
        (ontable box2 region2__0__0)
   ))

)
