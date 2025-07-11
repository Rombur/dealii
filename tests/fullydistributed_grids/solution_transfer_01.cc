// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2021 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// Test distributed solution transfer with fullydistributed triangulations.

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "../grid/tests.h"


template <int dim>
class InterpolationFunction : public Function<dim>
{
public:
  InterpolationFunction()
    : Function<dim>(1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    return p.norm();
  }
};

template <int dim, typename TriangulationType>
void
test(TriangulationType &triangulation)
{
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(FE_Q<dim>(2));

  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  std::shared_ptr<Utilities::MPI::Partitioner> partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);

  VectorType vector(partitioner);

  VectorTools::interpolate(dof_handler, InterpolationFunction<dim>(), vector);
  vector.update_ghost_values();

  VectorType vector_loaded(partitioner);

  const std::string filename =
    "solution_transfer_" + std::to_string(dim) + "d_out";

  {
    SolutionTransfer<dim, VectorType> solution_transfer(dof_handler);
    solution_transfer.prepare_for_serialization(vector);

    triangulation.save(filename);
  }

  triangulation.clear();

  {
    triangulation.load(filename);
    dof_handler.distribute_dofs(FE_Q<dim>(2));

    SolutionTransfer<dim, VectorType> solution_transfer(dof_handler);
    solution_transfer.deserialize(vector_loaded);

    vector_loaded.update_ghost_values();
  }

  // Verify that error is 0.
  VectorType error(vector);
  error.add(-1, vector_loaded);

  deallog << (error.linfty_norm() < 1e-16 ? "PASSED" : "FAILED") << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  deallog.push("2d");
  {
    constexpr int dim = 2;

    parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(3);

    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(triangulation, MPI_COMM_WORLD);

    parallel::fullydistributed::Triangulation<dim> triangulation_pft(
      MPI_COMM_WORLD);
    triangulation_pft.create_triangulation(description);

    test<dim>(triangulation_pft);
  }
  deallog.pop();

  deallog.push("3d");
  {
    constexpr int dim = 3;

    parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(3);

    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(triangulation, MPI_COMM_WORLD);

    parallel::fullydistributed::Triangulation<dim> triangulation_pft(
      MPI_COMM_WORLD);
    triangulation_pft.create_triangulation(description);

    test<dim>(triangulation_pft);
  }
  deallog.pop();
}
