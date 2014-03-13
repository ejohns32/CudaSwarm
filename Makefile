NVFLAGS=-g -c -Xcompiler -O3

CFLAGS=-g -O3
IFLAGS=-I . -I/usr/local/cuda/include/
LDFLAGS=-L/opt/cuda/lib64 -L/usr/local/cuda-5.5/targets/x86_64-linux/lib/ -lstdc++ -lcudart -lm

GLCFLAGS=-ccbin g++ -m64
GLIFLAGS=-Icommon/inc
GLDFLAGS=-Lcommon/lib/linux/x86_64 -L/usr/lib64/nvidia -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW

all: consoleSwarm

swarmGraphics.o: swarmGraphics.cu swarmGraphics.h swarmAgent.h 
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

swarmQuad.o: swarmQuad.cu swarmQuad.h swarmAgent.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

swarmAgent.o: swarmAgent.cu swarmAgent.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

consoleSwarmDriver.o: swarmDriver.cu swarmDriver.h swarmAgent.h swarmQuad.h swarmGraphics.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DCONSOLE $(NVFLAGS) $(IFLAGS) $<

glSwarmDriver.o: swarmDriver.cu swarmDriver.h swarmAgent.h swarmQuad.h swarmGraphics.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

consoleSwarm.o: consoleSwarm.cu swarmDriver.h swarmAgent.h swarmQuad.h swarmGraphics.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) $(IFLAGS) $<

glSwarm.o: glSwarm.cu swarmDriver.h swarmAgent.h swarmQuad.h
	/usr/local/cuda-5.5/bin/nvcc -o $@ -c $< $(GLCFLAGS) $(GLIFLAGS)

consoleSwarm: consoleSwarm.o consoleSwarmDriver.o swarmAgent.o swarmQuad.o swarmGraphics.o
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS)

glSwarm: glSwarm.o glSwarmDriver.o swarmAgent.o swarmQuad.o
	/usr/local/cuda-5.5/bin/nvcc -o $@ $^ $(GLCFLAGS) $(GLDFLAGS)

clean:
	rm -f consoleSwarm glSwarm swarmAgent.o swarmQuad.o swarmGraphics.o consoleSwarmDriver.o glSwarmDriver.o consoleSwarm.o glSwarm.o
