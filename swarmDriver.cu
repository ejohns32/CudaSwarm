#include "swarmDriver.h"
#include <cfloat>

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

		return SwarmAgent(team, (hashRand(i)) * SwarmAgent::maxPosition().x, hashRand(i + total) * SwarmAgent::maxPosition().y, cos(2 * CUDART_PI_F * hashRand(i + 2 * total)), sin(2 * CUDART_PI_F * hashRand(i + 2 * total)));
	}
};

void swarmSetup(thrust::device_vector<SwarmAgent> &dSwarm, unsigned int numTeams, unsigned int numAgentsPerTeam)
{
	thrust::host_vector<SwarmAgent> hSwarm(numTeams * numAgentsPerTeam);

	thrust::tabulate(hSwarm.begin(), hSwarm.end(), SpawnTeam(numTeams, numAgentsPerTeam));

	dSwarm = hSwarm;
}

struct AgentUpdate {
	SubSwarm subSwarm;
	float timeStep;

	AgentUpdate(SubSwarm subSwarm, float timeStep) : subSwarm(subSwarm), timeStep(timeStep) {}

	__host__ __device__ static int sgn(float val) {
		return (val > 0) - (val < 0);
	}

	__host__ __device__ void operator()(SwarmAgent &agent) {
		if (agent.alive) {
			/*SwarmAgent *closestTeam = NULL;
			float closestTeamDist = VIEW_DISTANCE;*/
			SwarmAgent *closestEnemy = NULL;
			float closestEnemyDist = FLT_MAX;

			for (SwarmAgent *itr = subSwarm.begin(); itr != subSwarm.end(); ++itr)
			{
				if (itr != &agent && itr->alive)
				{
					float dist = itr->distance(agent.position.x, agent.position.y);
					/*if (itr->team == agent.team && dist < closestTeamDist)
					{
						closestTeamDist = dist;
						closestTeam = itr;
					} else*/ if (itr->team != agent.team && dist < closestEnemyDist) {
						closestEnemyDist = dist;
						closestEnemy = itr;
					}
				}
			}

			if (closestEnemy != NULL) {
				float xDif = closestEnemy->position.x - agent.position.x;
				float yDif = closestEnemy->position.y - agent.position.y;
				float angle = atan2f(yDif, xDif);

				// probably want gradual direction changes
				agent.velocity.x = cos(angle);
				agent.velocity.y = sin(angle);
			}
/*
			if (closestTeam != NULL) {
				float xDif = closestTeam->position.x - agent.position.x;
				float yDif = closestTeam->position.y - agent.position.y;

				if (sgn(xDif) == sgn(agent.velocity.x) && sgn(agent.velocity.x) == -sgn(closestTeam->velocity.x) && abs(xDif) < TEAM_DISTANCE) {
					agent.velocity.x = -agent.velocity.x;
				} else if (sgn(yDif) == sgn(agent.velocity.y) && sgn(agent.velocity.y) == -sgn(closestTeam->velocity.y) && abs(yDif) < TEAM_DISTANCE) {
					agent.velocity.y = -agent.velocity.y;
				}
			}*/

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

	__device__ void operator()(SwarmAgent &agent) {
		if (agent.alive) {
			bool gunnaDie = false;
			for (SwarmAgent *itr = subSwarm.begin(); itr != subSwarm.end(); ++itr)
			{
				gunnaDie |= itr != &agent && itr->alive && itr->team != agent.team && itr->distance(agent.position.x, agent.position.y) < 0.01f;
			}
			__syncthreads();
			agent.alive = !gunnaDie;
		}
	}
};

// not sure this is the best way to do this
void checkCollisions(QuadTree &quadTree, thrust::device_vector<SwarmAgent> &dSwarm)
{
	for (int nodeNum = 0; nodeNum < quadTree.getNodeCount(); ++nodeNum)
	{
		SubSwarm subSwarm = quadTree.getNodeSubSwarm(nodeNum);
		thrust::for_each(thrust::device_pointer_cast(subSwarm.begin()), thrust::device_pointer_cast(subSwarm.end()), AgentAlive(subSwarm));
	}
}

void swarmStep(thrust::device_vector<SwarmAgent> &dSwarm, QuadTree &quadTree, float timeStep)
{
	updateSwarm(quadTree, dSwarm, timeStep);
	quadTree.update();
	checkCollisions(quadTree, dSwarm);
}