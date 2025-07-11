// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2021 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// verify restrictions on level differences imposed by
// hp::Refinement::limit_p_level_difference()
//
// set the center cell to the highest p-level in a hyper_cross geometry
// and verify that all other cells comply to the level difference


#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/refinement.h>

#include <vector>

#include "../tests.h"


template <int dim>
void
test(const unsigned int fes_size, const unsigned int max_difference)
{
  Assert(fes_size > 0, ExcInternalError());
  Assert(max_difference > 0, ExcInternalError());

  // setup FE collection
  hp::FECollection<dim> fes;
  while (fes.size() < fes_size)
    fes.push_back(FE_Q<dim>(1));

  const unsigned int contains_fe_index = 0;
  const auto         sequence = fes.get_hierarchy_sequence(contains_fe_index);

  // setup cross-shaped mesh
  Triangulation<dim> tria;
  {
    std::vector<unsigned int> sizes(Utilities::pow(2, dim),
                                    static_cast<unsigned int>(
                                      (sequence.size() - 1) / max_difference));
    GridGenerator::hyper_cross(tria, sizes);
  }

  deallog << "ncells:" << tria.n_cells() << ", nfes:" << fes.size() << std::endl
          << "sequence:" << sequence << std::endl;

  DoFHandler<dim> dofh(tria);
  dofh.distribute_dofs(fes);

  // set center to highest p-level
  const auto center_cell = dofh.begin();
  Assert(center_cell->center() == Point<dim>(), ExcInternalError());

  center_cell->set_active_fe_index(sequence.back());
  const bool fe_indices_changed =
    hp::Refinement::limit_p_level_difference(dofh,
                                             max_difference,
                                             contains_fe_index);
  tria.execute_coarsening_and_refinement();

  (void)fe_indices_changed;
  Assert(fe_indices_changed, ExcInternalError());

  // display number of cells for each FE index
  std::vector<unsigned int> count(fes.size(), 0);
  for (const auto &cell : dofh.active_cell_iterators())
    count[cell->active_fe_index()]++;
  deallog << "fe count:" << count << std::endl;

  if constexpr (running_in_debug_mode())
    {
      // check each cell's active FE index by its distance from the center
      for (const auto &cell : dofh.active_cell_iterators())
        {
          const double distance =
            cell->center().distance(center_cell->center());
          const unsigned int expected_level =
            (sequence.size() - 1) -
            max_difference * static_cast<unsigned int>(std::round(distance));

          Assert(cell->active_fe_index() == sequence[expected_level],
                 ExcInternalError());
        }
    }

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();

  test<2>(4, 1);
  test<2>(8, 2);
  test<2>(12, 3);
}
