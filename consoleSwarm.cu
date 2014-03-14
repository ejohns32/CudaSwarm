#include <iostream>
#include <cstring>

#include "swarmDriver.h"

void printSwarm(const thrust::device_vector<SwarmAgent> &dSwarm, float elapsedTime)
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

	std::cout << "---- swarm size: " << agentCount << " elapsedTime: " << elapsedTime << " ----" << std::endl;

	for (int row = 0; row < H_MAX_POSITION.y; ++row) {
		for (int col = 0; col < H_MAX_POSITION.x; ++col) {
			std::cout << display[row][col];
		}
		std::cout << std::endl;
	}
}

void consoleLoop(thrust::device_vector<SwarmAgent> &dSwarm, QuadTree &quadTree, float timeStep)
{
	float elapsedTime = 0.0f;

	while(true)
	{
		swarmStep(dSwarm, quadTree, timeStep);
		printSwarm(dSwarm, elapsedTime);
		usleep(timeStep * 1000 * 1000);
		elapsedTime += timeStep;
	}
}

int main()
{
	thrust::device_vector<SwarmAgent> dSwarm = thrust::device_vector<SwarmAgent>();
	swarmSetup(dSwarm, 2, 32);
	QuadTree quadTree = QuadTree(dSwarm);
	consoleLoop(dSwarm, quadTree, 0.01f);

	return 0;
}