#include <iostream>
#include <cstring>

#include "swarmGraphics.h"

void drawSwarm(const thrust::device_vector<SwarmAgent> &dSwarm)
{
	thrust::host_vector<SwarmAgent> hSwarm = dSwarm;
	char display[H_MAX_POSITION.y][H_MAX_POSITION.x];

	memset(display, ' ', H_MAX_POSITION.y * H_MAX_POSITION.x);


	int agentCount = 0;
	for (thrust::host_vector<SwarmAgent>::iterator itr = hSwarm.begin(); itr != hSwarm.end(); ++itr) {
		SwarmAgent temp = *itr;
		if (temp.alive) {
			if (temp.position.x < H_MAX_POSITION.x && temp.position.y < H_MAX_POSITION.y) {
				display[(int)temp.position.y][(int)temp.position.x] = temp.team == 1 ? 'X' : 'O';
			}
			++agentCount;
		}
	}

	std::cout << "---- swarm size: " << agentCount << " time: " << time << " ----" << std::endl;

	for (int row = 0; row < H_MAX_POSITION.y; ++row) {
		for (int col = 0; col < H_MAX_POSITION.x; ++col) {
			std::cout << display[row][col];
		}
		std::cout << std::endl;
	}
}