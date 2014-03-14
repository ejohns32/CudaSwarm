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

void swarmSetup(thrust::device_vector<SwarmAgent> &dSwarm, unsigned int numTeams, unsigned int numAgentsPerTeam)
{
	thrust::host_vector<SwarmAgent> hSwarm(numTeams * numAgentsPerTeam);

	thrust::tabulate(hSwarm.begin(), hSwarm.end(), SpawnTeam(numTeams, numAgentsPerTeam));

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
      if(itr == index || agents[indices[itr + min]].alive == false){
         itr++;
         continue;
      }

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
   if (numFoes) {
      float xDif = agents[indices[closestFoe]].position.x - (agents[indices[me]].position.x);
      float yDif = agents[indices[closestFoe]].position.y - (agents[indices[me]].position.y);
      float angle = atan2f(yDif, xDif);

      // probably want gradual direction changes
      agents[indices[me]].velocity.x = cos(angle);
      agents[indices[me]].velocity.y = sin(angle);
   }/*
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
   int numLeaves = quadTree.leaves.size();

   dim3 dimGrid(1024, 1024);

   dim3 dimBlock(32, 1);
   doBATTLE<<<dimGrid, dimBlock>>>(leaves, indices, agents, numLeaves, timeStep);

}

__global__ void getHit(int2 *leaves, int *indices, SwarmAgent *agents, int numLeaves){
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

   __syncthreads();
   me = index + min;
   itr = 0;
   if(agents[indices[me]].alive == false) return;
   while (itr < range)
   {
      if(itr == index || agents[indices[itr + min]].alive == false){
         itr++;
         continue;
      }
      dist = agents[indices[me]].distance(agents[indices[itr + min]].position.x, agents[indices[itr + min]].position.y);
      if (itr < range && agents[indices[me]].team == agents[indices[itr + min]].team){
         if(dist <= 1.0f){
            numFriends++;
            if(dist < closestFriendDist) {
               closestFriendDist = dist;
               closestFriend = itr;
            }
         }
      } else if (itr < range){
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
         agents[indices[closestFoe + min]].alive = false;
      else
         agents[indices[me]].alive = false;
   }
   if(closestFriendDist < 0.3f){
      agents[indices[me]].alive = false;
      agents[indices[closestFriend + min]].alive = false;
   }
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

void swarmStep(thrust::device_vector<SwarmAgent> &dSwarm, QuadTree &quadTree, float timeStep)
{
	updateSwarm(quadTree, timeStep);
	checkCollisions(quadTree);
	//collectTheBodies(quadTree);
	quadTree.buildTree();
}
