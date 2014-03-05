#ifndef _CUDA_SWARM_H_
#define _CUDA_SWARM_H_

const float2 MIN_POSITION = {0.0f, 0.0f};
cosnt float2 MAX_POSITION = {40.0f, 80.0f};

struct SwarmAgent {
	float2 position;
	float2 velocity;
	uint8_t team;
	bool alive;
};

#endif