#ifndef _SWARM_QUAD_H_
#define _SWARM_QUAD_H_

#include <thrust/device_vector.h>

#include "swarmAgent.h"

class SubSwarm {
	SwarmAgent *mBegin;
	SwarmAgent *mEnd;

public:
	SubSwarm(SwarmAgent *mBegin, SwarmAgent *mEnd) : mBegin(mBegin), mEnd(mEnd) {}
	__host__ __device__ SwarmAgent *begin() const { return mBegin; }
	__host__ __device__ SwarmAgent *end() const { return mEnd; }
};

class QuadTree
{
	SubSwarm dSubSwarm;

public:
	QuadTree(thrust::device_vector<SwarmAgent> &dSwarm);
	void update();
	unsigned int getNodeCount();
	SubSwarm getNodeSubSwarm(unsigned int node);
};

#endif