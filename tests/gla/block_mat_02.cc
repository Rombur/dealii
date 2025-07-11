// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2013 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// test problem in distribute_sparsity_pattern (block version)
// now Trilinos and PETSc produce the same block matrix


#define USE_PETSC_LA

#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>

#include <iostream>
#include <vector>

#include "../tests.h"

#include "gla.h"

template <class LA, int dim>
void
test()
{
  unsigned int myid    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int numproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if (myid == 0)
    deallog << "numproc=" << numproc << std::endl;


  parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD,
    typename Triangulation<dim>::MeshSmoothing(
      Triangulation<dim>::smoothing_on_refinement |
      Triangulation<dim>::smoothing_on_coarsening));

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  const FESystem<dim> stokes_fe(FE_Q<dim>(1), 1, FE_DGP<dim>(0), 1);


  DoFHandler<dim>           stokes_dof_handler(triangulation);
  std::vector<unsigned int> stokes_sub_blocks(1 + 1, 0);
  stokes_sub_blocks[1] = 1;
  stokes_dof_handler.distribute_dofs(stokes_fe);
  DoFRenumbering::component_wise(stokes_dof_handler, stokes_sub_blocks);

  const std::vector<types::global_dof_index> stokes_dofs_per_block =
    DoFTools::count_dofs_per_fe_block(stokes_dof_handler, stokes_sub_blocks);

  const unsigned int n_u = stokes_dofs_per_block[0],
                     n_p = stokes_dofs_per_block[1];

  std::vector<IndexSet> stokes_partitioning, stokes_relevant_partitioning;

  IndexSet stokes_index_set = stokes_dof_handler.locally_owned_dofs();
  stokes_partitioning.push_back(stokes_index_set.get_view(0, n_u));
  stokes_partitioning.push_back(stokes_index_set.get_view(n_u, n_u + n_p));

  const IndexSet stokes_relevant_set =
    DoFTools::extract_locally_relevant_dofs(stokes_dof_handler);
  stokes_relevant_partitioning.push_back(stokes_relevant_set.get_view(0, n_u));
  stokes_relevant_partitioning.push_back(
    stokes_relevant_set.get_view(n_u, n_u + n_p));

  stokes_relevant_set.print(deallog.get_file_stream());

  AffineConstraints<double> cm(stokes_index_set, stokes_relevant_set);

  DoFTools::make_hanging_node_constraints(stokes_dof_handler, cm);

  // boundary conditions ?
  cm.close();


  BlockDynamicSparsityPattern sp(stokes_relevant_partitioning);

  Table<2, DoFTools::Coupling> coupling(1 + 1, 1 + 1);
  coupling[0][0] = DoFTools::always;
  coupling[1][1] = DoFTools::always;
  coupling[0][1] = DoFTools::always;

  DoFTools::make_sparsity_pattern(stokes_dof_handler,
                                  coupling,
                                  sp,
                                  cm,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    MPI_COMM_WORLD));

  SparsityTools::distribute_sparsity_pattern(
    sp,
    stokes_dof_handler.locally_owned_dofs(),
    MPI_COMM_WORLD,
    stokes_relevant_set);


  sp.compress();

  typename LA::MPI::BlockSparseMatrix stokes_matrix;
  stokes_matrix.reinit(stokes_partitioning, sp, MPI_COMM_WORLD);

  stokes_matrix.print(deallog.get_file_stream());


  // done
  if (myid == 0)
    deallog << "OK" << std::endl;
}

template <int dim>
void
test_LA_Trilinos()
{
  using LA = LA_Trilinos;

  unsigned int myid    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int numproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if (myid == 0)
    deallog << "numproc=" << numproc << std::endl;


  parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD,
    typename Triangulation<dim>::MeshSmoothing(
      Triangulation<dim>::smoothing_on_refinement |
      Triangulation<dim>::smoothing_on_coarsening));

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  const FESystem<dim> stokes_fe(FE_Q<dim>(1), 1, FE_DGP<dim>(0), 1);


  DoFHandler<dim>           stokes_dof_handler(triangulation);
  std::vector<unsigned int> stokes_sub_blocks(1 + 1, 0);
  stokes_sub_blocks[1] = 1;
  stokes_dof_handler.distribute_dofs(stokes_fe);
  DoFRenumbering::component_wise(stokes_dof_handler, stokes_sub_blocks);

  const std::vector<types::global_dof_index> stokes_dofs_per_block =
    DoFTools::count_dofs_per_fe_block(stokes_dof_handler, stokes_sub_blocks);

  const unsigned int n_u = stokes_dofs_per_block[0],
                     n_p = stokes_dofs_per_block[1];

  std::vector<IndexSet> stokes_partitioning, stokes_relevant_partitioning;

  IndexSet stokes_index_set = stokes_dof_handler.locally_owned_dofs();
  stokes_partitioning.push_back(stokes_index_set.get_view(0, n_u));
  stokes_partitioning.push_back(stokes_index_set.get_view(n_u, n_u + n_p));

  const IndexSet stokes_relevant_set =
    DoFTools::extract_locally_relevant_dofs(stokes_dof_handler);
  stokes_relevant_partitioning.push_back(stokes_relevant_set.get_view(0, n_u));
  stokes_relevant_partitioning.push_back(
    stokes_relevant_set.get_view(n_u, n_u + n_p));
  stokes_relevant_set.print(deallog.get_file_stream());

  AffineConstraints<double> cm(stokes_index_set, stokes_relevant_set);

  DoFTools::make_hanging_node_constraints(stokes_dof_handler, cm);

  // boundary conditions ?
  cm.close();
  TrilinosWrappers::BlockSparsityPattern
    // typename LA::MPI::CompressedBlockSparsityPattern
    sp(stokes_partitioning, MPI_COMM_WORLD);


  Table<2, DoFTools::Coupling> coupling(1 + 1, 1 + 1);

  coupling[0][0] = DoFTools::always;
  coupling[1][1] = DoFTools::always;
  coupling[0][1] = DoFTools::always;

  DoFTools::make_sparsity_pattern(stokes_dof_handler,
                                  coupling,
                                  sp,
                                  cm,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    MPI_COMM_WORLD));

  sp.compress();

  // sp.print(deallog.get_file_stream());

  LA::MPI::BlockSparseMatrix stokes_matrix;
  stokes_matrix.reinit(sp);

  stokes_matrix.print(deallog.get_file_stream());


  // done
  if (myid == 0)
    deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    log;

  {
    deallog.push("PETSc");
    test<LA_PETSc, 2>();
    deallog.pop();
    deallog.push("Trilinos");
    test<LA_Trilinos, 2>();
    deallog.pop();
    deallog.push("TrilinosAlt");
    test_LA_Trilinos<2>();
    deallog.pop();
  }

  // compile, don't run
  // if (myid==9999)
  //  test<LA_Dummy>();
}
