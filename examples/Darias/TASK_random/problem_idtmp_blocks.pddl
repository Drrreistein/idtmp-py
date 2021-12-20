;14:45:33 19/12

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
          box0 - block
          box1 - block
          box2 - block
          box3 - block
          -1__0__0__0 - direction
          0__-1__0__0 - direction
          0__0__1__0 - direction
          0__1__0__0 - direction
          1__0__0__0 - direction
   )

   (:init
          (handempty)
          (ontable box3 region0__0__0)
          (ontable box0 region0__0__0)
          (ontable box1 region1__0__0)
          (ontable box2 region1__0__0)
   )

   (:goal
   (and
        (handempty)
        (ontable box3 region1__0__0)
        (ontable box0 region1__0__0)
   ))

)
