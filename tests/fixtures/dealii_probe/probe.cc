// Does VectorTools::point_value evaluate at a NON-nodal point, and at what order?
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include <cstdio>
#include <cmath>
using namespace dealii;
class U : public Function<2> {
public: double value(const Point<2>&p, const unsigned int=0) const override {
  return p[0]*(1-p[0])*p[1]*(1-p[1]); } };
int main(){
  for (unsigned n : {8u,16u,32u}) {
    Triangulation<2> tria; GridGenerator::hyper_cube(tria,0,1);
    tria.refine_global(std::log2(n));
    FE_Q<2> fe(1); DoFHandler<2> dh(tria); dh.distribute_dofs(fe);
    Vector<double> sol(dh.n_dofs());
    VectorTools::interpolate(dh, U(), sol);
    U ex; double worst=0.0;
    for (unsigned i=0;i<44;i++) for (unsigned j=0;j<44;j++) {
      Point<2> p((i+0.5)/44.0,(j+0.5)/44.0);          // deliberately NON-nodal
      double got = VectorTools::point_value(dh, sol, p);
      worst = std::max(worst, std::fabs(got - ex.value(p)));
    }
    std::printf("N=%3u  NDOF = %u  max|probe-exact| = %.6e\n", n, dh.n_dofs(), worst);
  }
}
