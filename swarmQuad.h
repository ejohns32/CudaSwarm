#ifndef _SWARM_QUAD_H_
#define _SWARM_QUAD_H_

#include <thrust/device_vector.h>

#include "swarmCuda.h"

class QuadTree
{
	thrust::device_vector<SwarmAgent> &dSwarm;

public:
	QuadTree(thrust::device_vector<SwarmAgent> &dSwarm) : dSwarm(dSwarm) {}
	void update();
	thrust::device_vector<SwarmAgent> getNearby(SwarmAgent agent);
};

#endif