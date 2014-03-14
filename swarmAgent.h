#ifndef _SWARM_AGENT_H_
#define _SWARM_AGENT_H_

__constant__ int2 D_MAX_POSITION = {30, 30};
const int2 H_MAX_POSITION = {30, 30};

struct SwarmAgent {
	float2 position;
	float2 velocity;
	uint8_t team;
	bool alive;

	__device__ __host__ SwarmAgent() : team(0), alive(false) {
	   position.x = 0; position.y = 0;
	   velocity.x = 0; velocity.y = 0;
   }
   __device__ __host__ SwarmAgent(uint8_t team, float xPos, float yPos, float xVel, float yVel) : team(team), alive(true) {
	position.x = xPos; position.y = yPos;
	velocity.x = xVel; velocity.y = yVel;
}


	__device__ __host__ static int2 maxPosition()
	{
#ifdef __CUDA_ARCH__
return D_MAX_POSITION;
#else
return H_MAX_POSITION;
#endif
	}
	
	__host__ __device__ void update(const float timeStep) {		position.x += velocity.x * timeStep;
		position.y += velocity.y * timeStep;

		if (position.x >= maxPosition().x) {
			position.x -= position.x - maxPosition().x;
			velocity.x = -velocity.x;
		} else if (position.x < 0) {
			position.x = -position.x;
			velocity.x = -velocity.x;
		}

		if (position.y >= maxPosition().y) {
			position.y -= position.y - maxPosition().y;
			velocity.y = -velocity.y;
		} else if (position.y < 0) {
			position.y = -position.y;
			velocity.y = -velocity.y;
		}
	}

	__host__ __device__ float distance(const float x, const float y) const {
		float difx = position.x - x;
		float dify = position.y - y;
		return sqrt(difx * difx + dify * dify);
	}
};


#endif
