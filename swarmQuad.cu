#include "swarmQuad.h"

void QuadTree::update(thrust::device_vector<SwarmAgent> &dSwarm)
{
	tempSave = dSwarm;// nothing for now
}

thrust::device_vector<SwarmAgent> QuadTree::getNearby(const SwarmAgent agent)
{
	thrust::device_vector<SwarmAgent> rtnValue = thrust::device_vector<SwarmAgent>();

	// should be done with thrust::copyif() in update
	for (thrust::device_vector<SwarmAgent>::iterator itr = tempSave.begin(); itr != tempSave.end(); ++itr)
	{
		SwarmAgent temp = *itr;
		if (temp.distance(agent.position.x, agent.position.y) < VIEW_DISTANCE)
		{
			rtnValue.push_back(*itr);
		}
	}

	return rtnValue;
}