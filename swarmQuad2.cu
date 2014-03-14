#include "swarmQuad2.h"

// Utility functions to encode leaves and children in single int
// are defined in util.h:
//   bool is_empty(int id);
//   bool is_node(int id);
//   bool is_leaf(int id);
//   int get_empty_id();
//   int get_leaf_id(int offset);
//   int get_leaf_offset(int id);
//   int child_tag_mask(int tag, int which_child, int level, int max_level);

bbox compute_bounding_box(const thrust::device_vector<SwarmAgent> &points)
{
  return thrust::reduce(points.begin(), points.end(), bbox(), merge_bboxes());
}

void compute_tags(const thrust::device_vector<SwarmAgent> &points, const bbox &bounds, int max_level, thrust::device_vector<int> &tags)
{
  thrust::transform(points.begin(), 
                    points.end(), 
                    tags.begin(), 
                    classify_point(bounds, max_level));
}


void sort_points_by_tag(thrust::device_vector<int> &tags, thrust::device_vector<int> &indices)
{
  thrust::sequence(indices.begin(), indices.end());
  thrust::sort_by_key(tags.begin(), tags.end(), indices.begin());
}


void compute_child_tag_masks(const thrust::device_vector<int> &active_nodes,
                             int level,
                             int max_level,
                             thrust::device_vector<int> &children)
{
  // For each active node, generate the tag mask for each of its 4 children
  thrust::tabulate(children.begin(), children.end(),
                   child_index_to_tag_mask(level, max_level, active_nodes.data()));
}


void find_child_bounds(const thrust::device_vector<int> &tags,
                       const thrust::device_vector<int> &children,
                       int level,
                       int max_level,
                       thrust::device_vector<int> &lower_bounds,
                       thrust::device_vector<int> &upper_bounds)
{
  // Locate lower and upper bounds for points in each quadrant
  thrust::lower_bound(tags.begin(),
                      tags.end(),
                      children.begin(),
                      children.end(),
                      lower_bounds.begin());
  
  int length = (1 << (max_level - level) * 2) - 1;

  using namespace thrust::placeholders;

  thrust::upper_bound(tags.begin(),
                      tags.end(),
                      thrust::make_transform_iterator(children.begin(), _1 + length),
                      thrust::make_transform_iterator(children.end(), _1 + length),
                      upper_bounds.begin());
}


void classify_children(const thrust::device_vector<int> &lower_bounds,
                       const thrust::device_vector<int> &upper_bounds,
                       int level,
                       int max_level,
                       int threshold,
                       thrust::device_vector<int> &child_node_kind)
{
  thrust::transform(lower_bounds.begin(), lower_bounds.end(),
                    upper_bounds.begin(),
                    child_node_kind.begin(),
                    classify_node(threshold, level == max_level));
}


std::pair<int,int> enumerate_nodes_and_leaves(const thrust::device_vector<int> &child_node_kind,
                                              thrust::device_vector<int> &nodes_on_this_level,
                                              thrust::device_vector<int> &leaves_on_this_level)
{
  // Enumerate nodes at this level
  thrust::transform_exclusive_scan(child_node_kind.begin(), 
                                   child_node_kind.end(), 
                                   nodes_on_this_level.begin(), 
                                   is_a<NODE>(), 
                                   0, 
                                   thrust::plus<int>());
  
  // Enumerate leaves at this level
  thrust::transform_exclusive_scan(child_node_kind.begin(), 
                                   child_node_kind.end(), 
                                   leaves_on_this_level.begin(), 
                                   is_a<LEAF>(), 
                                   0, 
                                   thrust::plus<int>());

  std::pair<int,int> num_nodes_and_leaves_on_this_level;

  num_nodes_and_leaves_on_this_level.first = nodes_on_this_level.back() + (child_node_kind.back() == NODE ? 1 : 0);
  num_nodes_and_leaves_on_this_level.second = leaves_on_this_level.back() + (child_node_kind.back() == LEAF ? 1 : 0);

  return num_nodes_and_leaves_on_this_level;
}

void create_child_nodes(const thrust::device_vector<int> &child_node_kind,
                        const thrust::device_vector<int> &nodes_on_this_level,
                        const thrust::device_vector<int> &leaves_on_this_level,
                        int num_leaves,
                        thrust::device_vector<int> &nodes)
{
  int num_children = child_node_kind.size();

  int children_begin = nodes.size();
  nodes.resize(nodes.size() + num_children);
  
  thrust::transform(thrust::make_zip_iterator(
                        thrust::make_tuple(
                            child_node_kind.begin(), nodes_on_this_level.begin(), leaves_on_this_level.begin())),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            child_node_kind.end(), nodes_on_this_level.end(), leaves_on_this_level.end())),
                    nodes.begin() + children_begin,
                    write_nodes(nodes.size(), num_leaves));
}





void create_leaves(const thrust::device_vector<int> &child_node_kind,
                   const thrust::device_vector<int> &leaves_on_this_level,
                   const thrust::device_vector<int> &lower_bounds,
                   const thrust::device_vector<int> &upper_bounds,
                   int num_leaves_on_this_level,
                   thrust::device_vector<int2> &leaves)
{
  int children_begin = leaves.size();

  leaves.resize(leaves.size() + num_leaves_on_this_level);

  thrust::scatter_if(thrust::make_transform_iterator(
                         thrust::make_zip_iterator(
                             thrust::make_tuple(lower_bounds.begin(), upper_bounds.begin())),
                         make_leaf()),
                     thrust::make_transform_iterator(
                         thrust::make_zip_iterator(
                             thrust::make_tuple(lower_bounds.end(), upper_bounds.end())),
                         make_leaf()),
                     leaves_on_this_level.begin(),
                     child_node_kind.begin(),
                     leaves.begin() + children_begin,
                     is_a<LEAF>());
} 


void activate_nodes_for_next_level(const thrust::device_vector<int> &children,
                                   const thrust::device_vector<int> &child_node_kind,
                                   int num_nodes_on_this_level,
                                   thrust::device_vector<int> &active_nodes)
{
  // Set active nodes for the next level to be all the childs nodes from this level
  active_nodes.resize(num_nodes_on_this_level);
  
  thrust::copy_if(children.begin(),
                  children.end(),
                  child_node_kind.begin(),
                  active_nodes.begin(),
                  is_a<NODE>());
}

void QuadTree::buildTree(){
  tags.clear();
  leaves.clear();
  nodes.clear();
  indices.clear();
  
  const int num_points = agents.size();

  /******************************************
   * 1. Compute bounding box                *
   ******************************************/

  bounds = compute_bounding_box(agents);

  /******************************************
   * 2. Classify points                     *
   ******************************************/

  tags.resize(num_points, 0);
  
  compute_tags(agents, bounds, maxLevel, tags);
  /******************************************
   * 3. Sort according to classification    *
   ******************************************/

  indices.resize(num_points, 0);

  sort_points_by_tag(tags, indices);
  /******************************************
   * 4. Build the tree                      *
   ******************************************/

  thrust::device_vector<int> active_nodes(1,0);

  // Build the tree one level at a time, starting at the root
  for (int level = 1 ; !active_nodes.empty() && level <= maxLevel ; ++level)
  {
    /******************************************
     * 5. Calculate children                  *
     ******************************************/

    // New children: 4 quadrants per active node = 4 children
    thrust::device_vector<int> children(4*active_nodes.size());
    compute_child_tag_masks(active_nodes, level, maxLevel, children);

    /******************************************
     * 6. Determine interval for each child   *
     ******************************************/

    // For each child we need interval bounds
    thrust::device_vector<int> lower_bounds(children.size());
    thrust::device_vector<int> upper_bounds(children.size());
    find_child_bounds(tags, children, level, maxLevel, lower_bounds, upper_bounds);

    /******************************************
     * 7. Mark each child as empty/leaf/node  *
     ******************************************/

    // Mark each child as either empty, a node, or a leaf
    thrust::device_vector<int> child_node_kind(children.size(), 0);
    classify_children(lower_bounds, upper_bounds, level, maxLevel, threshold, child_node_kind);

    /******************************************
     * 8. Enumerate nodes and leaves          *
     ******************************************/

    // Enumerate the nodes and leaves at this level
    thrust::device_vector<int> leaves_on_this_level(child_node_kind.size());
    thrust::device_vector<int> nodes_on_this_level(child_node_kind.size());

    // Enumerate nodes and leaves at this level
    std::pair<int,int> num_nodes_and_leaves_on_this_level =
      enumerate_nodes_and_leaves(child_node_kind, nodes_on_this_level, leaves_on_this_level);
 
    /******************************************
     * 9. Add the children to the node list   *
     ******************************************/

    create_child_nodes(child_node_kind, nodes_on_this_level, leaves_on_this_level, leaves.size(), nodes);

    /******************************************
     * 10. Add the leaves to the leaf list     *
     ******************************************/

    create_leaves(child_node_kind, leaves_on_this_level, lower_bounds, upper_bounds, num_nodes_and_leaves_on_this_level.second, leaves);
 
    /******************************************
     * 11. Set the nodes for the next level    *
     ******************************************/

    activate_nodes_for_next_level(children, child_node_kind, num_nodes_and_leaves_on_this_level.first, active_nodes);
  }
}

unsigned int QuadTree::getNodeCount()
{
	return leaves.size();
}

SubSwarm QuadTree::getNodeSubSwarm(unsigned int idx)
{
   int2 dblInt = leaves[idx];
   int start = dblInt.x, end = dblInt.y;
	return SubSwarm(thrust::raw_pointer_cast(indices.data() + start),
             thrust::raw_pointer_cast(indices.data() + end));
}
