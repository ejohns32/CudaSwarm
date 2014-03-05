#ifndef _SWARM_GRAPHICS_H_
#define _SWARM_GRAPHICS_H_

#include <thrust/device_vector.h>

#include "swarmCuda.h"

void drawSwarm(const thrust::device_vector<SwarmAgent> &dSwarm, const float time);

#endif