;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
;;Setting seed to 1229
(define (problem unpack)

	(:domain blocks)

	(:objects
		s_block  - block
		m_block  - block
		h_block  - block
		region_1_0_0  - location
		region_1_0_1  - location
		region_1_0_-1  - location
		region_1_-1_1  - location
		region_2_0_0  - location
		region_2_-1_1  - location
	)
  (:init
		(ontable s_block region_1_0_0)
		(ontable m_block region_1_0_1)
		(ontable h_block region_1_0_-1)
		(clear s_block)
		(clear m_block)
		(clear h_block)
        (handempty)
        (not (clear region_1_0_0))
        (not (clear region_1_0_1))
        (not (clear region_1_0_-1))
		(clear region_1_-1_1)
		(clear region_2_0_0)
		(clear region_2_-1_1)
	)

	(:goal
	(and
		(or
			(ontable s_block region_2_0_0)
			(ontable s_block region_2_-1_1)
		)
		(handempty)
	)
		
	)
)


