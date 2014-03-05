#include "swarmQuad.h"

QuadTree::QuadTree(thrust::device_vector<SwarmAgent> &dSwarm) : dSubSwarm(thrust::raw_pointer_cast(dSwarm.data()), thrust::raw_pointer_cast(dSwarm.data() + dSwarm.size())) {}

void QuadTree::update()
{
	// rebuild the tree
}

unsigned int QuadTree::getNodeCount()
{
	return 1;
}

SubSwarm QuadTree::getNodeSubSwarm(unsigned int node)
{
	return dSubSwarm;
}