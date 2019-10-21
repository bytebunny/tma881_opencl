CC := gcc
CFLAGS := -std=c11 -O3 -Wall -flto #-march=native -ffast-math
LIBS := -L/usr/lib64/ -lm -lOpenCL

OBJS := heat_diffusion.o 

.PHONY: all test clean
all: heat_diffusion 

# Rule to generate object files:
heat_diffusion: $(OBJS) compute_next_temp.cl reduce.cl compute_diff.cl
	$(CC) -o $@ $(OBJS) $(CFLAGS) $(LIBS)

test_opencl: test_opencl.o dot_prod_mul.cl
	$(CC) -o $@ test_opencl.o $(CFLAGS) $(LIBS)

$(OBJS) : heat_diffusion.h

test:
	tar -czvf heat_diffusion.tar.gz heat_diffusion.c heat_diffusion.h Makefile
	./check_submission.py heat_diffusion.tar.gz
clean:
	rm -rvf *.o heat_diffusion test_opencl extracted/ heat_diffusion.tar.gz vgcore*
