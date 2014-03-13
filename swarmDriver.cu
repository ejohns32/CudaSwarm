#include "swarmDriver.h"

void swarmLoop(thrust::device_vector<SwarmAgent> &dSwarm, float timeStep)
{
	QuadTree quadTree = QuadTree(dSwarm);

	float time = 0.0f;

	while(true)
	{
		swarmStep(dSwarm, quadTree, timeStep);
		usleep(timeStep * 1000 * 1000);
		time += timeStep;
	}
}

int main()
{
	thrust::device_vector<SwarmAgent> dSwarm = thrust::device_vector<SwarmAgent>();
	setup(dSwarm);
	swarmLoop(dSwarm, 0.1f);

	return 0;
}