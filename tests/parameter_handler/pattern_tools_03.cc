// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2017 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#include <deal.II/base/numbers.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>

#include <boost/core/demangle.hpp>

#include <memory>

#include "../tests.h"

using namespace Patterns::Tools;

// Try conversion on container types

template <class T>
void
test(T t)
{
  auto p = Convert<T>::to_pattern();
  deallog << "Pattern  : " << p->description() << std::endl;
  auto s = Convert<T>::to_string(t);
  deallog << "To String: " << s << std::endl;
  deallog << "To value : " << Convert<T>::to_string(Convert<T>::to_value(s))
          << std::endl;
}

int
main()
{
  initlog();

  int           t0 = 1;
  unsigned int  t1 = 2;
  unsigned char t2 = 3;
  std::string   t3 = "Ciao";
  double        t4 = 4.0;

  std::vector<int>           t10(2, t0);
  std::vector<unsigned int>  t11(2, t1);
  std::vector<unsigned char> t12(2, t2);
  std::vector<std::string>   t13(2, t3);
  std::vector<double>        t14(2, t4);

  test(t10);
  test(t11);
  test(t12);
  test(t13);
  test(t14);

  return 0;
}
