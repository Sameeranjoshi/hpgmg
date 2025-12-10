hpgmg-fv-y.c += $(call thisdir, \
	timers.c \
	level.c \
	operators.fv2.c \
	mg.c \
	solvers.c \
	hpgmg-fv.c \
	)
hpgmg-fv-y.cu += $(call thisdir, \
	cuda/operators.fv2.cu \
	)
