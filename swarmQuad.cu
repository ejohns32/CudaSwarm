#include "swarmQuad.h"

void QuadTree::update()
{
	// rebuld the tree
}

thrust::device_vector<SwarmAgent> QuadTree::getNearby(const SwarmAgent agent)
{
	thrust::device_vector<SwarmAgent> rtnValue = thrust::device_vector<SwarmAgent>();

	// should be done with thrust::copyif() in update
	for (thrust::device_vector<SwarmAgent>::iterator itr = dSwarm.begin(); itr != dSwarm.end(); ++itr)
	{
		SwarmAgent temp = *itr;
		if (temp.distance(agent.position.x, agent.position.y) < VIEW_DISTANCE)
		{
			rtnValue.push_back(*itr);
		}
	}

	return rtnValue;
}