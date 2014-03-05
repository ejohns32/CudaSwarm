NVFLAGS=-g -c -Xcompiler -O3
CFLAGS=-g -O3
LDFLAGS=-L/opt/cuda/lib64 -L/usr/local/cuda-5.5/targets/x86_64-linux/lib/ -lstdc++ -lcudart -lm

all: gpu_swarm

swarmGraphics.o: swarmGraphics.cu swarmGraphics.h swarmAgent.h 
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) -I . $<

swarmQuad.o: swarmQuad.cu swarmQuad.h swarmAgent.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) -I . $<

swarmAgent.o: swarmAgent.cu swarmAgent.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) -I . $<

swarmDriver.o: swarmDriver.cu swarmAgent.h swarmQuad.h swarmGraphics.h
	nvcc -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA $(NVFLAGS) -I . $<

gpu_swarm: swarmDriver.o swarmAgent.o swarmQuad.o swarmGraphics.o
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f gpu_swarm swarmAgent.o swarmQuad.o swarmGraphics.o swarmDriver.o
