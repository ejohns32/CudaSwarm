#ifndef _SWARM_QUAD_H_
#define _SWARM_QUAD_H_

#include <thrust/device_vector.h>

#include "swarmAgent.h"

inline __device__ __host__
bool is_empty(int id) { return id == 0xffffffff; }

inline __device__ __host__
bool is_node(int id) { return id > 0; }

inline __device__ __host__
bool is_leaf(int id) { return id < 0; }

inline __device__ __host__
int get_empty_id() { return 0xffffffff; }

inline __device__ __host__
int get_leaf_id(int offset) { return 0x80000000 | offset; }

inline __device__ __host__
int get_leaf_offset(int id) { return 0x80000000 ^ id; }

inline __device__ __host__
int child_tag_mask(int tag, int which_child, int level, int max_level)
{
  int shift = (max_level - level) * 2;
  return tag | (which_child << shift);
}

// Markers
enum { NODE = 1, LEAF = 2, EMPTY = 4 };

template <int CODE>
struct is_a
{
  typedef int result_type;
  inline __device__ __host__
  int operator()(int code) { return code == CODE ? 1 : 0; }
};

struct bbox
{
  float xmin, xmax;
  float ymin, ymax;

  inline __host__ __device__
  bbox() : xmin(FLT_MAX), xmax(-FLT_MAX), ymin(FLT_MAX), ymax(-FLT_MAX)
  {}
  
  inline __host__ __device__
  bbox(const SwarmAgent &p) : xmin(p.position.x), xmax(p.position.x),
     ymin(p.position.y), ymax(p.position.y)
  {}
};

class SubSwarm {
	SwarmAgent *mBegin;
	SwarmAgent *mEnd;

public:
	SubSwarm(SwarmAgent *mBegin, SwarmAgent *mEnd) : mBegin(mBegin), mEnd(mEnd) {}
	__host__ __device__ SwarmAgent *begin() const { return mBegin; }
	__host__ __device__ SwarmAgent *end() const { return mEnd; }
};

__host__ __device__
int point_to_tag(const SwarmAgent &p, bbox box, int max_level)
{
  int result = 0;
  
  for (int level = 1 ; level <= max_level ; ++level)
  {
    // Classify in x-direction
    float xmid = 0.5f * (box.xmin + box.xmax);
    int x_hi_half = (p.position.x < xmid) ? 0 : 1;
  
    // Push the bit into the result as we build it
    result |= x_hi_half;
    result <<= 1;
  
    // Classify in y-direction
    float ymid = 0.5f * (box.ymin + box.ymax);
    int y_hi_half = (p.position.y < ymid) ? 0 : 1;
  
    // Push the bit into the result as we build it
    result |= y_hi_half;
    result <<= 1;
  
    // Shrink the bounding box, still encapsulating the point
    box.xmin = (x_hi_half) ? xmid : box.xmin;
    box.xmax = (x_hi_half) ? box.xmax : xmid;
    box.ymin = (y_hi_half) ? ymid : box.ymin;
    box.ymax = (y_hi_half) ? box.ymax : ymid;
  }
  // Unshift the last
  result >>= 1;

  return result;
}

class QuadTree
{
	thrust::device_vector<SwarmAgent> agents;
   thrust::device_vector<int> nodes;
   thrust::device_vector<int2> leaves;
   thrust::device_vector<int> tags;
   thrust::device_vector<int> indices;
   bbox bounds;
   int maxLevel;
   int threshold;

public:
	QuadTree(thrust::device_vector<SwarmAgent> &dSwarm,
      int max_level, int thresh): agents(dSwarm)
   {
      maxLevel = max_level;
      threshold = threshold;
   }
   void buildTree();
	unsigned int getNodeCount();
	SubSwarm getNodeSubSwarm(unsigned int node);
};

#endif
