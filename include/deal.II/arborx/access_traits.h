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

#ifndef dealii_arborx_access_traits_h
#define dealii_arborx_access_traits_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_ARBORX
#  include <deal.II/base/bounding_box.h>

#  include <ArborX.hpp>


DEAL_II_NAMESPACE_OPEN

/**
 *  Structure that encapsulates a vector of dealii::Points used as queries.
 */
struct PointIntersect
{
  /**
   * Constructor.
   */
  template <int dim, typename Number>
  PointIntersect(std::vector<dealii::Point<dim, Number>> const &dim_points)
  {
    static_assert(dim != 1, "dim equal to one is not supported.");

    unsigned int const size = dim_points.size();
    points.reserve(size);
    for (unsigned int i = 0; i < size; ++i)
      {
        points.emplace_back(static_cast<float>(dim_points[i][0]),
                            static_cast<float>(dim_points[i][1]),
                            dim == 2 ? 0. :
                                       static_cast<float>(dim_points[i][2]));
      }
  }

  /**
   * Number of Point stored in the structure.
   */
  std::size_t
  size() const
  {
    return points.size();
  }

  std::vector<dealii::Point<3, float>> points;
};



/**
 * Structure that encapsulates a vector of dealii::BoundingBox used as
 * queries.
 */
struct BoundingBoxIntersect
{
  /**
   * Constructor.
   */
  template <int dim, typename Number>
  BoundingBoxIntersect(std::vector<dealii::BoundingBox<dim, Number>> const &bb)
  {
    unsigned int const size = bb.size();
    bounding_boxes.reserve(size);
    dealii::Point<3, float> min_corner_arborx(0., 0., 0.);
    dealii::Point<3, float> max_corner_arborx(0., 0., 0.);
    for (unsigned int i = 0; i < size; ++i)
      {
        auto boundary_points                  = bb[i].get_boundary_points();
        dealii::Point<dim, Number> min_corner = boundary_points.first;
        dealii::Point<dim, Number> max_corner = boundary_points.second;
        for (int d = 0; d < dim; ++d)
          {
            min_corner_arborx[d] = static_cast<float>(min_corner[d]);
            max_corner_arborx[d] = static_cast<float>(max_corner[d]);
          }
        bounding_boxes.emplace_back(
          std::make_pair(min_corner_arborx, max_corner_arborx));
      }
  }

  /**
   * Number of Point stored in the structure.
   */
  std::size_t
  size() const
  {
    return bounding_boxes.size();
  }

  std::vector<dealii::BoundingBox<3, float>> bounding_boxes;
};

DEAL_II_NAMESPACE_CLOSE

namespace ArborX
{
  /**
   * This struct allows ArborX to use std::vector<dealii::BoundingBox> as
   * primitive.
   */
  template <int dim, typename Number>
  struct AccessTraits<std::vector<dealii::BoundingBox<dim, Number>>,
                      PrimitivesTag>
  {
    static_assert(dim != 1, "dim equal to one is not supported.");

    /**
     * Return the size of the vector @p v.
     */
    static std::size_t
    size(std::vector<dealii::BoundingBox<dim, Number>> const &v)
    {
      return v.size();
    }

    /**
     * Return an ArborX::Box from the dealii::BoundingBox v[i].
     */
    static Box
    get(std::vector<dealii::BoundingBox<dim, Number>> const &v, std::size_t i)
    {
      auto                       boundary_points = v[i].get_boundary_points();
      dealii::Point<dim, Number> min_corner      = boundary_points.first;
      dealii::Point<dim, Number> max_corner      = boundary_points.second;
      // ArborX assumes that the bounding box coordinates use float and that the
      // bounding box is 3D
      return {{static_cast<float>(min_corner[0]),
               static_cast<float>(min_corner[1]),
               dim == 2 ? 0 : static_cast<float>(min_corner[2])},
              {static_cast<float>(max_corner[0]),
               static_cast<float>(max_corner[1]),
               dim == 2 ? 0. : static_cast<float>(max_corner[2])}};
    }

    using memory_space = Kokkos::HostSpace;
  };



  /**
   * This struct allows ArborX to use std::vector<dealii::Point> as
   * primitive.
   */
  template <int dim, typename Number>
  struct AccessTraits<std::vector<dealii::Point<dim, Number>>, PrimitivesTag>
  {
    static_assert(dim != 1, "dim equal to one is not supported.");

    /**
     * Return the size of the vector @p v.
     */
    static std::size_t
    size(std::vector<dealii::Point<dim, Number>> const &v)
    {
      return v.size();
    }

    /**
     * Return an ArborX::Point from the dealii::Point v[i].
     */
    static Point
    get(std::vector<dealii::Point<dim, Number>> const &v, std::size_t i)
    {
      // ArborX assumes that the point coordinates use float and that the point
      // is 3D
      return {{static_cast<float>(v[i][0]),
               static_cast<float>(v[i][1]),
               dim == 2 ? 0 : static_cast<float>(v[i][2])}};
    }

    using memory_space = Kokkos::HostSpace;
  };



  /**
   * This struct allows ArborX to use PointIntersect in a predicate.
   */
  template <>
  struct AccessTraits<dealii::PointIntersect, PredicatesTag>
  {
    /**
     * Number of Point stored in @p pt_intersect.
     */
    static std::size_t
    size(dealii::PointIntersect const &pt_intersect)
    {
      return pt_intersect.size();
    }

    /**
     * Return an Arbox::intersects(ArborX::Point) object constructed from the
     * ith dealii::Point stored in @p pt_intersect.
     */
    static auto
    get(dealii::PointIntersect const &pt_intersect, std::size_t i)
    {
      return intersects(Point{pt_intersect.points[i][0],
                              pt_intersect.points[i][1],
                              pt_intersect.points[i][2]});
    }

    using memory_space = Kokkos::HostSpace;
  };



  /**
   * This struct allows ArborX to use BoundingBoxIntersect in a predicate.
   */
  template <>
  struct AccessTraits<dealii::BoundingBoxIntersect, PredicatesTag>
  {
    /**
     * Number of BoundingBox stored in @p bb_intersect.
     */
    static std::size_t
    size(dealii::BoundingBoxIntersect const &bb_intersect)
    {
      return bb_intersect.size();
    }

    /**
     * Return an Arbox::intersects(ArborX::Box) object constructed from the
     * ith dealii::BoundingBox stored in @p bb_intersect.
     */
    static auto
    get(dealii::BoundingBoxIntersect const &bb_intersect, std::size_t i)
    {
      auto boundary_points =
        bb_intersect.bounding_boxes[i].get_boundary_points();
      dealii::Point<3, float> min_corner = boundary_points.first;
      dealii::Point<3, float> max_corner = boundary_points.second;

      return intersects(Box{{min_corner[0], min_corner[1], min_corner[2]},
                            {max_corner[0], max_corner[1], max_corner[2]}});
    }

    using memory_space = Kokkos::HostSpace;
  };
} // namespace ArborX

#endif

#endif
