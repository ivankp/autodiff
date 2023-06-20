#include <iostream>
using std::cout, std::endl;

#include "autodiff.hh"
using ivan::autodiff::var;
using ivan::autodiff::mkvar;

#include "debug.hh"

int main(int argc, char** argv) {
  std::ios_base::sync_with_stdio(false);
  // if (argc != 2) {
  //   std::cerr << "usage: " << argv[0] << " arg\n";
  //   return 1;
  // }

  // var<double,0> x;
  // auto y = mkvar<1>(3.);
  //
  // var tmp(7.);
  //
  // cout << tmp << '\n';
  // cout << y.d[0] << '\n';
  //
  // // test_type< merge_var_t< var<int,0,2>, var<double,1,2> > >{};
  // // test_type< merge_var_t< var<int,0,2>, double > >{};
  //
  // TEST(( index< 2, var<int,0,2> > ))
  // TEST(( index< 2, int > ))

  var<double,0> x(3);
  var<double,1> y(5);

  TEST( d<1>(y) )

  TEST(x)
  TEST(x.d[0])

  TEST(y)
  TEST(y.d[0])

  TEST((x*y))
  TEST((x*y).d[0])
  TEST((x*y).d[1])

  TEST(((x*y)+y))
  TEST(((x*y)+y).d[0])
  TEST(((x*y)+y).d[1])

  TEST(( (x+1)*y ))
  TEST(( (x+1)*y ).d[0])
  TEST(( (x+1)*y ).d[1])
}
