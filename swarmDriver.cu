#include <unistd.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <math_constants.h>
#include <cfloat>
#include "swarmDriver.h"

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
	uint32_t numPerTeam;
	uint32_t total;

	SpawnTeam(uint8_t numTeams, uint32_t numPerTeam) : numTeams(numTeams), numPerTeam(numPerTeam), total(numTeams * numPerTeam) {}

	__host__ __device__ SwarmAgent operator()(unsigned int i) {
		uint8_t team = i / numPerTeam;

		return SwarmAgent(team, (hashRand(i)) * SwarmAgent::maxPosition().x, hashRand(i + total) * SwarmAgent::maxPosition().y, cos(2 * CUDART_PI_F * hashRand(i + 2 * total)), sin(2 * CUDART_PI_F * hashRand(i + 2 * total)));
	}
};

void swarmSetup(thrust::device_vector<SwarmAgent> &dSwarm, QuadTree &quadTree, unsigned int numTeams, unsigned int numAgentsPerTeam)
{
	int maxPosition = sqrt(numTeams * numAgentsPerTeam);
	SwarmAgent::setMaxPosition(maxPosition);

	dSwarm.resize(numTeams * numAgentsPerTeam);

	thrust::tabulate(dSwarm.begin(), dSwarm.end(), SpawnTeam(numTeams, numAgentsPerTeam));

    quadTree.setMaxLevel(1 + log(numTeams * numAgentsPerTeam / 32) / log(4));
}

__host__ __device__ static int sgn(float val) {
   return (val > 0) - (val < 0);
}

struct TeamLeafStats
{
	int numFriends, numFoes, closestFriend, closestFoe;
   float closestFriendDist, closestEnemyDist;

   __host__ __device__ TeamLeafStats() : numFriends(0), numFoes(0), closestFriend(0), closestFoe(0), closestFriendDist(FLT_MAX), closestEnemyDist(FLT_MAX) {}
};

__host__ __device__ TeamLeafStats findClosest(int *indices, SwarmAgent *agents, int me, int min, int max)
{
	TeamLeafStats stats;

   for (int itr = min; itr < max; itr++)
   {
      if(itr == me || agents[indices[itr]].alive == false) {
         continue;
      }

      float dist = agents[indices[me]].distance(agents[indices[itr]].position.x, agents[indices[itr]].position.y);
      if (agents[indices[me]].team == agents[indices[itr]].team){
         stats.numFriends++;
         if(dist < stats.closestFriendDist) {
            stats.closestFriendDist = dist;
            stats.closestFriend = itr;
         }
      } else {
         stats.numFoes++;
         if(dist < stats.closestEnemyDist) {
            stats.closestEnemyDist = dist;
            stats.closestFoe = itr;
         }
      }
   }

   return stats;
}

__host__ __device__ void calcVelocity(int *indices, SwarmAgent *agents, int me, TeamLeafStats stats)
{
   if (stats.numFoes) {
      float xDif = agents[indices[stats.closestFoe]].position.x - (agents[indices[me]].position.x);
      float yDif = agents[indices[stats.closestFoe]].position.y - (agents[indices[me]].position.y);
      float angle = atan2f(yDif, xDif);

      // probably want gradual direction changes
      agents[indices[me]].velocity.x = cos(angle);
      agents[indices[me]].velocity.y = sin(angle);
   } 
   /*if (numFriends) {
      float xDif = agents[indices[closestFriend]].position.x - agents[indices[me]].position.x;
      float yDif = agents[indices[closestFriend]].position.y - agents[indices[me]].position.y;

      if (sgn(xDif) == sgn(agents[indices[me]].velocity.x) && abs(xDif) < TEAM_DISTANCE) {
         agents[indices[me]].velocity.x = -agents[indices[me]].velocity.x;
      } else if (sgn(yDif) == sgn(agents[indices[me]].velocity.y) && abs(yDif) < TEAM_DISTANCE) {
         agents[indices[me]].velocity.y = -agents[indices[me]].velocity.y;
      }
   }*/
}

__global__ void doBATTLE(int2 *leaves, int *indices, SwarmAgent *agents, int numLeaves, float timeStep){
   
   int leafIdx = gridDim.x * blockIdx.y + blockIdx.x;
   int index, me;
   int min, max, range;

   if(leafIdx >= numLeaves)
      return;

   min = leaves[leafIdx].x;
   max = leaves[leafIdx].y;

   range = max - min;
   index = threadIdx.y * blockDim.x + threadIdx.x;
   
   if(index >= range)
      return;

  	me = min + index;

  	if(!agents[indices[me]].alive)
  		return;

  	TeamLeafStats stats = findClosest(indices, agents, me, min, max);
  	__syncthreads();
  	calcVelocity(indices, agents, me, stats);
   __syncthreads();
   agents[indices[me]].update(timeStep);
}

void updateSwarm(QuadTree &quadTree, float timeStep)
{
   int2 *leaves = thrust::raw_pointer_cast(quadTree.leaves.data());
   int *indices = thrust::raw_pointer_cast(quadTree.indices.data());
   SwarmAgent *agents = thrust::raw_pointer_cast(quadTree.agents.data());
   int numLeaves = quadTree.leaves.size();

   dim3 dimGrid((numLeaves / 1024) + ((numLeaves % 1024) ? 1 : 0), 1024);
   dim3 dimBlock(32, 4);
   doBATTLE<<<dimGrid, dimBlock>>>(leaves, indices, agents, numLeaves, timeStep);

}

__host__ __device__ void killStuff(int *indices, SwarmAgent *agents, int me, TeamLeafStats stats) {
	if(stats.numFoes > 0 && stats.closestEnemyDist < 0.3f){
      if(stats.numFriends >= stats.numFoes)
         agents[indices[stats.closestFoe]].alive = false;
      else
         agents[indices[me]].alive = false;
   }
   /*if(stats.numFriends > 0 && stats.closestFriendDist < 0.3f){
      agents[indices[me]].alive = false;
      agents[indices[stats.closestFriend]].alive = false;
   }*/
}

__global__ void getHit(int2 *leaves, int *indices, SwarmAgent *agents, int numLeaves){
   int leafIdx = gridDim.x * blockIdx.y + blockIdx.x;
   int index, me;
   int min, max, range;

   if(leafIdx >= numLeaves)
      return;

   min = leaves[leafIdx].x;
   max = leaves[leafIdx].y;

   range = max - min;
   index = threadIdx.y * blockDim.x + threadIdx.x;
   
   if(index >= range)
      return;

   me = index + min;
   if(agents[indices[me]].alive == false) return;
   
   TeamLeafStats stats = findClosest(indices, agents, me, min, max);
   __syncthreads();
   killStuff(indices, agents, me, stats);
   
}

// not sure this is the best way to do this
void checkCollisions(QuadTree &quadTree)
{
   int2 *leaves = thrust::raw_pointer_cast(quadTree.leaves.data());
   int *indices = thrust::raw_pointer_cast(quadTree.indices.data());
   SwarmAgent *agents = thrust::raw_pointer_cast(quadTree.agents.data());

   int numLeaves = quadTree.leaves.size();
   
   dim3 dimGrid((numLeaves / 1024) + ((numLeaves % 1024) ? 1 : 0), 1024);

   dim3 dimBlock(32, 4);
   getHit<<<dimGrid, dimBlock>>>(leaves, indices, agents, numLeaves);
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
    while(tree.agents.size() / 32 < pow(4, tree.getMaxLevel() - 1)){
       tree.setMaxLevel(tree.getMaxLevel() - 1);
       std::cout << "Tree Levels: " << tree.getMaxLevel() << "\n";
    }
}

struct otherTeam
{
   __host__ __device__
      bool operator()(const SwarmAgent &agent)
      {
         return agent.team == 0;
      }
};

unsigned int swarmStep(thrust::device_vector<SwarmAgent> &dSwarm, QuadTree &quadTree, float timeStep)
{
	updateSwarm(quadTree, timeStep);
	checkCollisions(quadTree);
	collectTheBodies(quadTree);
	quadTree.buildTree();

   return thrust::count_if(dSwarm.begin(), dSwarm.end(), otherTeam());

}
