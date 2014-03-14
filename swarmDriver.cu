#include <unistd.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <math_constants.h>
#include "swarmAgent.h"
#include "swarmQuad2.h"
#include "swarmGraphics.h"
const int NUM_TEAMS = 2;
const int NUM_AGENTS_PER_TEAM = 32;

const float TEAM_DISTANCE = 2;

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

__host__ __device__ static int sgn(float val) {
   return (val > 0) - (val < 0);
}

__global__ void doBATTLE(int2 *leaves, int *indices, SwarmAgent *agents, int numLeaves, float timeStep){
   
   int leafIdx = gridDim.x * blockIdx.y + blockIdx.x;
   int index, me;
   int numFriends = 0, numFoes = 0, min, max, range;
   int closestFriend = 0, closestFoe = 0, itr = 0;
   float dist, closestFriendDist = FLT_MAX, closestEnemyDist = FLT_MAX;


   if(leafIdx >= numLeaves)
      return;

   min = leaves[leafIdx].x;
   max = leaves[leafIdx].y;

   range = max - min;
   index = threadIdx.y * blockDim.x + threadIdx.x;
   
   if(index >= range)
      return;
   me = index + min;
   itr = 0;
   while (itr < range)
   {
      if(itr == index)
         itr++;
      __syncthreads();

      dist = agents[indices[me]].distance(agents[indices[itr + min]].position.x, agents[indices[itr + min]].position.y);
      if (itr < range && agents[indices[me]].team == agents[indices[itr + min]].team){
         numFriends++;
         if(dist < closestFriendDist) {
            closestFriendDist = dist;
            closestFriend = itr + min;
         }
      } else if (itr < range){
         numFoes++;
         if(dist < closestEnemyDist) {
            closestEnemyDist = dist;
            closestFoe = itr + min;
         }
      }
      itr++;
   }
   __syncthreads();
/*   if (numFoes) {
      float xDif = agents[indices[closestFoe]].position.x - agents[indices[me]].position.x;
      float yDif = agents[indices[closestFoe]].position.y - agents[indices[me]].position.y;
      float angle = atan2f(yDif, xDif);

      // probably want gradual direction changes
      agents[indices[me]].velocity.x = cos(angle);
      agents[indices[me]].velocity.y = sin(angle);
   }
   if (numFriends) {
      float xDif = agents[indices[closestFriend]].position.x - agents[indices[me]].position.x;
      float yDif = agents[indices[closestFriend]].position.y - agents[indices[me]].position.y;

      if (sgn(xDif) == sgn(agents[indices[me]].velocity.x) && abs(xDif) < TEAM_DISTANCE) {
         agents[indices[me]].velocity.x = -agents[indices[me]].velocity.x;
      } else if (sgn(yDif) == sgn(agents[indices[me]].velocity.y) && abs(yDif) < TEAM_DISTANCE) {
         agents[indices[me]].velocity.y = -agents[indices[me]].velocity.y;
      }
   }*/
   __syncthreads();
   agents[indices[me]].update(timeStep);
}


void updateSwarm(QuadTree &quadTree, float timeStep)
{
   int2 *leaves = thrust::raw_pointer_cast(quadTree.leaves.data());
   int *indices = thrust::raw_pointer_cast(quadTree.indices.data());
   SwarmAgent *agents = thrust::raw_pointer_cast(quadTree.agents.data());
   SwarmAgent agent;
   SwarmAgent agent2;
   int numLeaves = quadTree.leaves.size();

   dim3 dimGrid(1024, 1024);

   dim3 dimBlock(32, 32);
   doBATTLE<<<dimGrid, dimBlock>>>(leaves, indices, agents, numLeaves, timeStep);
   agent = quadTree.agents[0];
   cudaMemcpy((void *)(&agent2), (void *)agents, sizeof(SwarmAgent), cudaMemcpyDeviceToHost);
   std::cout << agent.position.x << " " << agent.position.y;

}
/*
struct AgentAlive {
	SubSwarm subSwarm;

	AgentAlive(SubSwarm subSwarm) : subSwarm(subSwarm) {}

	__host__ __device__ void operator()(SwarmAgent &agent) {
		if (agent.alive) {
			for (SwarmAgent *itr = subSwarm.begin(); itr != subSwarm.end(); ++itr)
			{
				if (itr != &agent && itr->alive && (int)itr->position.x == (int)agent.position.x && (int)itr->position.y == (int)agent.position.y)
				{
					agent.alive = false; // race condition with itr->alive in if (feature-bug: only one dies, so there's a winner)
				}
			}
		}
	}
};*/
__global__ void getHit(int2 *leaves, int *indices, SwarmAgent *agents, int numLeaves){
   int leafIdx = 1024 * blockIdx.y + blockIdx.x;
   int index;
   SwarmAgent *me;
   int numFriends = 0, numFoes = 0, diff;
   int closestFriend = 0, closestFoe = 0, itr = 0;
   float dist, closestFriendDist = FLT_MAX, closestEnemyDist = FLT_MAX;

   if(leafIdx >= numLeaves)
      return;

   index = leaves[leafIdx].x + (threadIdx.y * 32) +threadIdx.x;
   
   if(index > (diff = leaves[leafIdx].y - leaves[leafIdx].x))
      return;

   __shared__ SwarmAgent agentsOnLeaf[128];
   __syncthreads();
   memcpy(agentsOnLeaf + index, &(agents[indices[index]]), sizeof(SwarmAgent));

   me = &(agentsOnLeaf[index]);

   while (itr < diff)
   {
      if(itr == index)
         itr++;
      __syncthreads();

      if (itr < diff && me->team == agentsOnLeaf[itr].team){
         if(dist <= 1.0f){
            numFriends++;
            if(dist < closestFriendDist) {
               closestFriendDist = dist;
               closestFriend = itr;
            }
         }
      } else if (itr < diff){
         if(dist <= 1.0f){
            numFoes++;
            if(dist < closestEnemyDist) {
               closestEnemyDist = dist;
               closestFoe = itr;
            }
         }
      }
      itr++;
   }
   __syncthreads();
   if(closestEnemyDist < 0.3f){
      if(numFriends >= numFoes)
         agentsOnLeaf[closestFoe].alive = false;
      else
         me->alive = false;
   }
   if(closestFriendDist < 0.3f){
      me->alive = false;
      agentsOnLeaf[closestFriend].alive = false;
   }
   memcpy(&(agents[indices[index]]), me, sizeof(SwarmAgent));
}

struct is_dead
{
   __host__ __device__
      bool operator()(const SwarmAgent &agent)
      {
         if(agent.alive)
            return false;
         return true;
      }
};

void collectTheBodies(QuadTree &tree){
    tree.agents.erase(thrust::remove_if(tree.agents.begin(), tree.agents.end(), is_dead()), tree.agents.end());
}

// not sure this is the best way to do this
void checkCollisions(QuadTree &quadTree)
{
   int2 *leaves = thrust::raw_pointer_cast(quadTree.leaves.data());
   int *indices = thrust::raw_pointer_cast(quadTree.indices.data());
   SwarmAgent *agents = thrust::raw_pointer_cast(quadTree.agents.data());

   int numLeaves = quadTree.leaves.size();
   
   dim3 dimGrid(1024, (numLeaves / 1024) + ((numLeaves % 1024) ? 1 : 0));
   

   dim3 dimBlock(32, 4);
   getHit<<<dimGrid, dimBlock>>>(leaves, indices, agents, numLeaves);
}

void swarmLoop(thrust::device_vector<SwarmAgent> &dSwarm, float timeStep)
{
	QuadTree quadTree = QuadTree(dSwarm, 10, 32);
   quadTree.buildTree();

	float time = 0.0f;

	while(true)
	{
		updateSwarm(quadTree, timeStep);
		//checkCollisions(quadTree);
      //collectTheBodies(quadTree);
   	drawSwarm(dSwarm, time);
      quadTree.buildTree();
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
