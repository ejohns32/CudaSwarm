#include <unistd.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "swarmAgent.h"

SwarmAgent::SwarmAgent() : team(0), alive(false) {
	position.x = 0; position.y = 0;
	velocity.x = 0; velocity.y = 0;
}

SwarmAgent::SwarmAgent(uint8_t team, float xPos, float yPos, float xVel, float yVel) : team(team), alive(true) {
	position.x = xPos; position.y = yPos;
	velocity.x = xVel; velocity.y = yVel;
}
