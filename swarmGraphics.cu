#include <iostream>
#include <cstring>

#include "swarmGraphics.h"

void drawSwarm(const thrust::device_vector<SwarmAgent> &dSwarm) {
	thrust::host_vector<SwarmAgent> hSwarm = dSwarm;
	char display[MAX_POSITION.y][MAX_POSITION.x];

	memset(display, ' ', MAX_POSITION.y * MAX_POSITION.x);

	for (thrust::host_vector<SwarmAgent>::iterator itr = hSwarm.begin(); itr != hSwarm.end(); ++itr) {
		SwarmAgent temp = *itr;
		display[(int)temp.position.y][(int)temp.position.x] = temp.team == 1 ? 'X' : 'O';
	}

	for (int row = 0; row < MAX_POSITION.y; ++row) {
		for (int col = 0; col < MAX_POSITION.x; ++col) {
			std::cout << display[row][col];
		}
		std::cout << std::endl;
	}
}