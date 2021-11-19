;01:47:45 19/11

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
          region_shelf__-1__-2 - location
          region_shelf__-1__-3 - location
          region_shelf__-1__0 - location
          region_shelf__-1__1 - location
          region_shelf__-1__2 - location
          region_shelf__-1__3 - location
          region_shelf__-2__-1 - location
          region_shelf__-2__-2 - location
          region_shelf__-2__-3 - location
          region_shelf__-2__0 - location
          region_shelf__-2__1 - location
          region_shelf__-2__2 - location
          region_shelf__-2__3 - location
          region_shelf__-3__-1 - location
          region_shelf__-3__-2 - location
          region_shelf__-3__-3 - location
          region_shelf__-3__0 - location
          region_shelf__-3__1 - location
          region_shelf__-3__2 - location
          region_shelf__-3__3 - location
          region_shelf__0__-1 - location
          region_shelf__0__-2 - location
          region_shelf__0__-3 - location
          region_shelf__0__0 - location
          region_shelf__0__1 - location
          region_shelf__0__2 - location
          region_shelf__0__3 - location
          region_shelf__1__-1 - location
          region_shelf__1__-2 - location
          region_shelf__1__-3 - location
          region_shelf__1__0 - location
          region_shelf__1__1 - location
          region_shelf__1__2 - location
          region_shelf__1__3 - location
          region_shelf__2__-1 - location
          region_shelf__2__-2 - location
          region_shelf__2__-3 - location
          region_shelf__2__0 - location
          region_shelf__2__1 - location
          region_shelf__2__2 - location
          region_shelf__2__3 - location
          region_shelf__3__-1 - location
          region_shelf__3__-2 - location
          region_shelf__3__-3 - location
          region_shelf__3__0 - location
          region_shelf__3__1 - location
          region_shelf__3__2 - location
          region_shelf__3__3 - location
          region_table__-1__-1 - location
          region_table__-1__-2 - location
          region_table__-1__-3 - location
          region_table__-1__0 - location
          region_table__-1__1 - location
          region_table__-1__2 - location
          region_table__-1__3 - location
          region_table__-2__-1 - location
          region_table__-2__-2 - location
          region_table__-2__-3 - location
          region_table__-2__0 - location
          region_table__-2__1 - location
          region_table__-2__2 - location
          region_table__-2__3 - location
          region_table__-3__-1 - location
          region_table__-3__-2 - location
          region_table__-3__-3 - location
          region_table__-3__0 - location
          region_table__-3__1 - location
          region_table__-3__2 - location
          region_table__-3__3 - location
          region_table__0__-1 - location
          region_table__0__-2 - location
          region_table__0__-3 - location
          region_table__0__0 - location
          region_table__0__1 - location
          region_table__0__2 - location
          region_table__0__3 - location
          region_table__1__-1 - location
          region_table__1__-2 - location
          region_table__1__-3 - location
          region_table__1__0 - location
          region_table__1__1 - location
          region_table__1__2 - location
          region_table__1__3 - location
          region_table__2__-1 - location
          region_table__2__-2 - location
          region_table__2__-3 - location
          region_table__2__0 - location
          region_table__2__1 - location
          region_table__2__2 - location
          region_table__2__3 - location
          region_table__3__-1 - location
          region_table__3__-2 - location
          region_table__3__-3 - location
          region_table__3__0 - location
          region_table__3__1 - location
          region_table__3__2 - location
          region_table__3__3 - location
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
        (ontable box1 region_shelf__0__0)
   ))

)
