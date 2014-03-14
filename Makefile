NVFLAGS=-g -c -Xcompiler -O3 -arch compute_12

CFLAGS=-g -O3
IFLAGS=-I . -I/usr/local/cuda/include/
LDFLAGS=-L/opt/cuda/lib64 -L/usr/local/cuda-5.5/targets/x86_64-linux/lib/ -lstdc++ -lcudart -lm

GLCFLAGS=-ccbin g++ -m64
GLIFLAGS=-Icommon/inc
GLDFLAGS=-Lcommon/lib/linux/x86_64 -L/usr/lib64/nvidia -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW

all: consoleSwarm glSwarm

swarmQuad.o: swarmQuad2.cu swarmQuad2.h swarmAgent.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

swarmAgent.o: swarmAgent.cu swarmAgent.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

swarmDriver.o: swarmDriver.cu swarmDriver.h swarmAgent.h swarmQuad2.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

consoleSwarm.o: consoleSwarm.cu swarmDriver.h swarmAgent.h swarmQuad2.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

glSwarm.o: glSwarm.cu swarmDriver.h swarmAgent.h swarmQuad2.h
	/usr/local/cuda-5.5/bin/nvcc -o $@ -c $< $(GLCFLAGS) $(GLIFLAGS)

consoleSwarm: consoleSwarm.o swarmDriver.o swarmAgent.o swarmQuad.o
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS)

glSwarm: glSwarm.o swarmDriver.o swarmAgent.o swarmQuad.o
	/usr/local/cuda-5.5/bin/nvcc -o $@ $^ $(GLCFLAGS) $(GLDFLAGS)

clean:
	rm -f consoleSwarm glSwarm swarmAgent.o swarmQuad.o swarmDriver.o consoleSwarm.o glSwarm.o
