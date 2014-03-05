#include "cudaSwarm.h"
#include "swarmGraphics.h"

void setup(thrust::device_vector<SwarmAgent> &swarm) {

}

void swarmLoop(thrust::device_vector<SwarmAgent> &swarm) {
	QuadTree quadTree = QuadTree(swarm);

	while(true) {
		updateSwarm(quadTree, swarm);
		updateQuadTree(quadTree, swarm);
		checkCollisions(quadTree, swarm);
		drawSwarm(swarm);
	}
}

void main() {
	thrust::device_vector<SwarmAgent> swarm = thrust::device_vector();
	setup(swarm);
	swarmLoop(swarm);
}
