#include <unistd.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "swarmAgent.h"
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
			hSwarm[team * NUM_AGENTS_PER_TEAM + agent] = SwarmAgent(team, agent * 2.0f, team * 3.0f, 2.0f, 1.0f);
		}
	}

	dSwarm = hSwarm;
}

struct AgentUpdate {
	QuadTree quadTree;
	float timeStep;

	AgentUpdate(QuadTree quadTree, float timeStep) : quadTree(quadTree), timeStep(timeStep) {}

	__host__ __device__ void operator()(SwarmAgent &agent) {
		// update velocity based on quadTree
		agent.update(timeStep);
	}
};

void updateSwarm(QuadTree &quadTree, thrust::device_vector<SwarmAgent> &dSwarm, float timeStep)
{
	thrust::for_each(dSwarm.begin(), dSwarm.end(), AgentUpdate(quadTree, timeStep));
}

void swarmLoop(thrust::device_vector<SwarmAgent> &dSwarm, float timeStep)
{
	QuadTree quadTree = QuadTree(dSwarm);

	float time = 0.0f;

	while(true)
	{
		updateSwarm(quadTree, dSwarm, timeStep);
		quadTree.update();
		//checkCollisions(quadTree, dSwarm);
		drawSwarm(dSwarm, time);
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
