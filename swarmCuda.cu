#include <unistd.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "swarmCuda.h"
#include "swarmQuad.h"
#include "swarmGraphics.h"

const int NUM_TEAMS = 2;
const int NUM_AGENTS_PER_TEAM = 20;

void setup(thrust::device_vector<SwarmAgent> &dSwarm)
{
	thrust::host_vector<SwarmAgent> hSwarm(NUM_TEAMS * NUM_AGENTS_PER_TEAM);

	for (int team = 0; team < NUM_TEAMS; ++team)
	{
		for (int agent = 0; agent < NUM_AGENTS_PER_TEAM; ++agent)
		{
			hSwarm[team * NUM_AGENTS_PER_TEAM + agent] = SwarmAgent(team, agent * 2.0f, team * 3.0f);
		}
	}

	dSwarm = hSwarm;
}

void swarmLoop(thrust::device_vector<SwarmAgent> &dSwarm)
{
	QuadTree quadTree = QuadTree(dSwarm);

	while(true)
	{
		//updateSwarm(quadTree, dSwarm);
		quadTree.update(dSwarm);
		//checkCollisions(quadTree, dSwarm);
		drawSwarm(dSwarm);
		usleep(1000 * 500); // sleep half a second
	}
}

int main()
{
	thrust::device_vector<SwarmAgent> dSwarm = thrust::device_vector<SwarmAgent>();
	setup(dSwarm);
	swarmLoop(dSwarm);

	return 0;
}
