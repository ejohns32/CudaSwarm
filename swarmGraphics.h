#ifndef _SWARM_GRAPHICS_H_
#define _SWARM_GRAPHICS_H_

#include <thrust/device_vector.h>

#include "swarmAgent.h"

void drawSwarm(const thrust::device_vector<SwarmAgent> &dSwarm);

#endif