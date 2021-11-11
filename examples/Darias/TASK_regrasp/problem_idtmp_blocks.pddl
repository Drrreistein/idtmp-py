;16:52:25 11/11

(define (problem regrasp-block)
   (:domain blocks)

   (:objects
          region_drawer__-1__-1 - location
          region_drawer__-1__0 - location
          region_drawer__-1__1 - location
          region_drawer__0__-1 - location
          region_drawer__0__0 - location
          region_drawer__0__1 - location
          region_drawer__1__-1 - location
          region_drawer__1__0 - location
          region_drawer__1__1 - location
          region_shelf__-1__-1 - location
          region_shelf__-1__0 - location
          region_shelf__-1__1 - location
          region_shelf__0__-1 - location
          region_shelf__0__0 - location
          region_shelf__0__1 - location
          region_shelf__1__-1 - location
          region_shelf__1__0 - location
          region_shelf__1__1 - location
          region_table__-1__-1 - location
          region_table__-1__0 - location
          region_table__-1__1 - location
          region_table__-2__-1 - location
          region_table__-2__0 - location
          region_table__-2__1 - location
          region_table__0__-1 - location
          region_table__0__0 - location
          region_table__0__1 - location
          region_table__1__-1 - location
          region_table__1__0 - location
          region_table__1__1 - location
          region_table__2__-1 - location
          region_table__2__0 - location
          region_table__2__1 - location
          box1 - block
          -1__0__0__0 - direction
          0__-1__0__0 - direction
          0__0__1__0 - direction
          0__1__0__0 - direction
          1__0__0__0 - direction
   )

   (:init
          (handempty)
          (ontable box1 region_drawer__0__0)
   )

   (:goal
   (and
        (handempty)
        (ontable box1 region_shelf__1__0)
   ))

)
