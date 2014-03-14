#ifndef _SWARM_QUAD_H_
#define _SWARM_QUAD_H_

#include <thrust/sequence.h>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <algorithm>
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

bbox compute_bounding_box(const thrust::device_vector<SwarmAgent> &points);

struct classify_node
{
  int threshold;
  int last_level;
  
  classify_node(int threshold, int last_level) : threshold(threshold), last_level(last_level) {}

  inline __device__ __host__
  int operator()(int lower_bound, int upper_bound) const
  {
    int count = upper_bound - lower_bound;
    if (count == 0)
    {
      return EMPTY;
    }
    else if (last_level || count < threshold)
    {
      return LEAF;
    }
    else
    {
      return NODE;
    }
  }
};

// Operator which merges two bounding boxes.
struct merge_bboxes
{
  inline __host__ __device__
  bbox operator()(const bbox &b0, const bbox &b1) const
  {
    bbox bounds;
    bounds.xmin = min(b0.xmin, b1.xmin);
    bounds.xmax = max(b0.xmax, b1.xmax);
    bounds.ymin = min(b0.ymin, b1.ymin);
    bounds.ymax = max(b0.ymax, b1.ymax);
    return bounds;
  }
};

__host__ __device__
inline int point_to_tag(const SwarmAgent &p, bbox box, int max_level)
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

// Classify a point with respect to the bounding box.
struct classify_point
{
  bbox box;
  int max_level;

  // Create the classifier
  classify_point(const bbox &b, int lvl) : box(b), max_level(lvl) {}

  // Classify a point
  inline __device__ __host__
  int operator()(const SwarmAgent &p) { return point_to_tag(p, box, max_level); }
};

struct child_index_to_tag_mask
{
  int level, max_level;
  thrust::device_ptr<const int> nodes;
  
  child_index_to_tag_mask(int lvl, int max_lvl, thrust::device_ptr<const int> nodes) : level(lvl), max_level(max_lvl), nodes(nodes) {}
  
  inline __device__ __host__
  int operator()(int idx) const
  {
    int tag = nodes[idx/4];
    int which_child = (idx&3);
    return child_tag_mask(tag, which_child, level, max_level);
  }
};

struct write_nodes
{
  int num_nodes, num_leaves;

  write_nodes(int num_nodes, int num_leaves) : 
    num_nodes(num_nodes), num_leaves(num_leaves) 
  {}

  template <typename tuple_type>
  inline __device__ __host__
  int operator()(const tuple_type &t) const
  {
    int node_type = thrust::get<0>(t);
    int node_idx  = thrust::get<1>(t);
    int leaf_idx  = thrust::get<2>(t);

    if (node_type == EMPTY)
    {
      return get_empty_id();
    }
    else if (node_type == LEAF)
    {
      return get_leaf_id(num_leaves + leaf_idx);
    }
    else
    {
      return num_nodes + 4 * node_idx;
    }
  }
};

struct make_leaf
{
  typedef int2 result_type;
  template <typename tuple_type>
  inline __device__ __host__
  int2 operator()(const tuple_type &t) const
  {
    int x = thrust::get<0>(t);
    int y = thrust::get<1>(t);

    return make_int2(x, y);
  }
};

class SubSwarm {
	int *mBegin;
	int *mEnd;

public:
	SubSwarm(int *mBegin, int *mEnd) : mBegin(mBegin), mEnd(mEnd) {}
	__host__ __device__ int *begin() const { return mBegin; }
	__host__ __device__ int *end() const { return mEnd; }
};

class QuadTree
{
   public:
	   thrust::device_vector<SwarmAgent> &agents;
      thrust::device_vector<int> nodes;
      thrust::device_vector<int2> leaves;
      thrust::device_vector<int> tags;
      thrust::device_vector<int> indices;
      bbox bounds;
      int maxLevel;
      int threshold;
	   QuadTree(thrust::device_vector<SwarmAgent> &dSwarm,
         int thresh): agents(dSwarm), maxLevel(1), threshold(threshold) {}
      void buildTree();
      void setMaxLevel(int maxLevel) {this->maxLevel = maxLevel;}
      int getMaxLevel(){return maxLevel;}
	   unsigned int getNodeCount();
	   SubSwarm getNodeSubSwarm(unsigned int node);
};

#endif
