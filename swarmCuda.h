#ifndef _SWARM_CUDA_H_
#define _SWARM_CUDA_H_

const int2 MAX_POSITION = {80, 20};
const float VIEW_DISTANCE = 10;

struct SwarmAgent {
	float2 position;
	float2 velocity;
	uint8_t team;
	bool alive;

	SwarmAgent() : team(0), alive(false) {
		position.x = 0; position.y = 0;
		velocity.x = 0; velocity.y = 0;
	}

	SwarmAgent(uint8_t team, float x, float y) : team(team), alive(true) {
		position.x = x; position.y = y;
		velocity.x = 0; velocity.y = 0;
	}

	__host__ __device__
	float distance(float x, float y) {
		float difx = position.x - x;
		float dify = position.y - y;
		return sqrt(difx * difx + dify * dify);
	}
};

#endif