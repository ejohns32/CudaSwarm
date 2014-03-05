#include <unistd.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <math_constants.h>

#include "swarmAgent.h"
#include "swarmQuad.h"
#include "swarmGraphics.h"

const int NUM_TEAMS = 2;
const int NUM_AGENTS_PER_TEAM = 32;

const float VIEW_DISTANCE = 10;

// from fun_with_points
__host__ __device__
float hashRand(unsigned int x)
{
	x = (x+0x7ed55d16) + (x<<12);
	x = (x^0xc761c23c) ^ (x>>19);
	x = (x+0x165667b1) + (x<<5);
	x = (x+0xd3a2646c) ^ (x<<9);
	x = (x+0xfd7046c5) + (x<<3);
	x = (x^0xb55a4f09) ^ (x>>16);
	return float(x) / UINT_MAX;
}

struct SpawnTeam {
	uint8_t numTeams;
	uint16_t numPerTeam;
	uint32_t total;

	SpawnTeam(uint8_t numTeams, uint16_t numPerTeam) : numTeams(numTeams), numPerTeam(numPerTeam), total(numTeams * numPerTeam) {}

	__host__ __device__ SwarmAgent operator()(unsigned int i) {
		uint8_t team = i / numPerTeam;

		return SwarmAgent(team, (hashRand(i) + team) * SwarmAgent::maxPosition().x / numTeams, hashRand(i + total) * SwarmAgent::maxPosition().y, cos(2 * CUDART_PI_F * hashRand(i + 2 * total)), sin(2 * CUDART_PI_F * hashRand(i + 2 * total)));
	}
};

void setup(thrust::device_vector<SwarmAgent> &dSwarm)
{
	thrust::host_vector<SwarmAgent> hSwarm(NUM_TEAMS * NUM_AGENTS_PER_TEAM);

	thrust::tabulate(hSwarm.begin(), hSwarm.end(), SpawnTeam(NUM_TEAMS, NUM_AGENTS_PER_TEAM));

	dSwarm = hSwarm;
}

struct AgentUpdate {
	SubSwarm subSwarm;
	float timeStep;

	AgentUpdate(SubSwarm subSwarm, float timeStep) : subSwarm(subSwarm), timeStep(timeStep) {}

	__host__ __device__ void operator()(SwarmAgent &agent) {
		if (agent.alive) {
			for (SwarmAgent *itr = subSwarm.begin(); itr != subSwarm.end(); ++itr)
			{
				if (itr->alive && itr->distance(agent.position.x, agent.position.y) < VIEW_DISTANCE)
				{
					// change velocity
				}
			}

			agent.update(timeStep);
		}
	}
};

void updateSwarm(QuadTree &quadTree, thrust::device_vector<SwarmAgent> &dSwarm, float timeStep)
{
	for (int nodeNum = 0; nodeNum < quadTree.getNodeCount(); ++nodeNum)
	{
		SubSwarm subSwarm = quadTree.getNodeSubSwarm(nodeNum);
		thrust::for_each(thrust::device_pointer_cast(subSwarm.begin()), thrust::device_pointer_cast(subSwarm.end()), AgentUpdate(subSwarm, timeStep));
	}
}

struct AgentAlive {
	SubSwarm subSwarm;

	AgentAlive(SubSwarm subSwarm) : subSwarm(subSwarm) {}

	__host__ __device__ void operator()(SwarmAgent &agent) {
		if (agent.alive) {
			for (SwarmAgent *itr = subSwarm.begin(); itr != subSwarm.end(); ++itr)
			{
				if (itr != &agent && itr->alive && (int)itr->position.x == (int)agent.position.x && (int)itr->position.y == (int)agent.position.y)
				{
					agent.alive = false; // race condition with itr->alive in if
				}
			}
		}
	}
};

void checkCollisions(QuadTree &quadTree, thrust::device_vector<SwarmAgent> &dSwarm)
{
	for (int nodeNum = 0; nodeNum < quadTree.getNodeCount(); ++nodeNum)
	{
		SubSwarm subSwarm = quadTree.getNodeSubSwarm(nodeNum);
		thrust::for_each(thrust::device_pointer_cast(subSwarm.begin()), thrust::device_pointer_cast(subSwarm.end()), AgentAlive(subSwarm));
	}
}

void swarmLoop(thrust::device_vector<SwarmAgent> &dSwarm, float timeStep)
{
	QuadTree quadTree = QuadTree(dSwarm);

	float time = 0.0f;

	while(true)
	{
		updateSwarm(quadTree, dSwarm, timeStep);
		quadTree.update();
		checkCollisions(quadTree, dSwarm);
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
