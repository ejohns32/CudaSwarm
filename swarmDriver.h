#include <unistd.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <math_constants.h>

#include "swarmAgent.h"
#include "swarmQuad.h"
#include "swarmGraphics.h"

const int NUM_TEAMS = 2;
const int NUM_AGENTS_PER_TEAM = 128;

const float VIEW_DISTANCE = 10;
const float TEAM_DISTANCE = 2;

void swarmStep(thrust::device_vector<SwarmAgent> &dSwarm, QuadTree &quadTree, float timeStep);
void setup(thrust::device_vector<SwarmAgent> &dSwarm);