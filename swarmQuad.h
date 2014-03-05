#ifndef _SWARM_QUAD_H_
#define _SWARM_QUAD_H_

#include <thrust/device_vector.h>

#include "swarmCuda.h"

class QuadTree
{
	thrust::device_vector<SwarmAgent> tempSave;

public:
	QuadTree(thrust::device_vector<SwarmAgent> &dSwarm) { update(dSwarm); }
	void update(thrust::device_vector<SwarmAgent> &dSwarm);
	thrust::device_vector<SwarmAgent> getNearby(SwarmAgent agent);
};

#endif