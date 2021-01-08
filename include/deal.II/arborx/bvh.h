// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_arborx_bvh_h
#define dealii_arborx_bvh_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_ARBORX
#  include <deal.II/arborx/access_traits.h>

#  include <ArborX_LinearBVH.hpp>
#  include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

/**
 * This class implements a wrapper around ArborX::BVH. BVH stands from Bounding
 * Volume Hierarchy.
 *
 * From [Wikipedia](https://en.wikipedia.org/wiki/Bounding_Volume_Hierarchy):
 * <blockquote>
 * A bounding volume hierarchy (BVH) is a tree structure on a set of geometric
 * objects. All geometric objects are wrapped in bounding volumes that form the
 * leaf nodes of the tree. These nodes are then grouped as small sets and
 * enclosed within larger bounding volumes. These, in turn, are also grouped and
 * enclosed within other larger bounding volumes in a recursive fashion,
 * eventually resulting in a tree structure with a single bounding volume at the
 * top of the tree. Bounding volume hierarchies are used to support several
 * operations on sets of geometric objects efficiently, such as in collision
 * detection and ray tracing.
 * </blockquote>
 *
 * Because ArborX uses Kokkos, Kokkos needs to be initialized and finalized
 * before using this class.
 */
class BVH
{
public:
  /**
   * Constructor. Use a vector of BoundingBox @p bounding_boxes as primitives.
   */
  template <int dim, typename Number>
  BVH(const std::vector<BoundingBox<dim, Number>> &bounding_boxes)
    : bvh(Kokkos::DefaultHostExecutionSpace{}, bounding_boxes)
  {}

  /**
   * Return the indices of the BoundingBox that satisfies the @p queries.
   * Because @p queries can contain multiple queries the function return a pair
   * of indices and offsets.
   *
   * @code
   * auto [indices, offset] = bvh.query(my_queries);
   * @endcode
   *
   * The indices that satisfy the first query are given by:
   *
   * @code
   * std::vector<int> first_query_indices;
   * for (int i = offset[0]; i < offset[1]; ++i)
   *   first_query_indices.push_back(indices[i]);
   * @endcode
   */
  template <typename QueryType>
  std::pair<std::vector<int>, std::vector<int>>
  query(const QueryType &queries)
  {
    Kokkos::View<int *, Kokkos::HostSpace> indices("indices", 0);

    Kokkos::View<int *, Kokkos::HostSpace> offset("offset", 0);
    ArborX::query(
      bvh, Kokkos::DefaultHostExecutionSpace{}, queries, indices, offset);
    std::vector<int> indices_vector;
    indices_vector.insert(indices_vector.begin(),
                          indices.data(),
                          indices.data() + indices.extent(0));
    std::vector<int> offset_vector;
    offset_vector.insert(offset_vector.begin(),
                         offset.data(),
                         offset.data() + offset.extent(0));

    return {indices_vector, offset_vector};
  }

private:
  /**
   * Underlying ArborX object.
   */
  ArborX::BVH<Kokkos::HostSpace> bvh;
};

DEAL_II_NAMESPACE_CLOSE

#endif
#endif
