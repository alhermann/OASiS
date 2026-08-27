/* Transient heat conduction  du/dt - div(k grad u) = f  on ONE rectangular subdomain
 * using Crank-Nicolson time stepping.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

using namespace dealii;

class Sampled1D
{
public:
  Sampled1D(std::vector<double> ys, std::vector<double> vs)
    : ys_(std::move(ys)), vs_(std::move(vs)) {}

  double operator()(const double y) const
  {
    if (ys_.size() == 1 || y <= ys_.front()) return vs_.front();
    if (y >= ys_.back()) return vs_.back();
    const auto it = std::upper_bound(ys_.begin(), ys_.end(), y);
    const std::size_t i = std::distance(ys_.begin(), it);
    const double w = (y - ys_[i - 1]) / (ys_[i] - ys_[i - 1]);
    return (1.0 - w) * vs_[i - 1] + w * vs_[i];
  }
private:
  std::vector<double> ys_, vs_;
};

class InterfaceFunction : public Function<2>
{
public:
  explicit InterfaceFunction(const Sampled1D &s) : s_(s) {}
  virtual double value(const Point<2> &p, const unsigned int = 0) const override
  { return s_(p[1]); }
private:
  const Sampled1D &s_;
};

class SourceTermA : public Function<2>
{
public:
  virtual double value(const Point<2> &p, const unsigned int = 0) const override
  {
    double x = p[0], y = p[1], t = current_time;
    double exp_t2 = std::exp(t/2.0), result = 0.0;
    result += -384.0*t*x*x*x*y*y*y + 1184.0*t*x*x*x*y*y + 3808.0*t*x*x*x*y - 4736.0*t*x*x*x;
    result += -316.0*t*x*x*y*y*y - 199.0*t*x*x*y*y + 4307.0*t*x*x*y + 796.0*t*x*x;
    result += 5148.0*t*x*y*y*y - 14883.0*t*x*y*y + 3255.0*t*x*y + 2700.0*t*x;
    result += 1264.0*t*y*y*y + 796.0*t*y*y - 2060.0*t*y;
    result += -768.0*x*x*x*y*y*y + 2368.0*x*x*x*y*y - 1600.0*x*x*x*y;
    result += -632.0*x*x*y*y*y - 398.0*x*x*y*y + 1030.0*x*x*y;
    result += 1080.0*x*y*y*y - 1350.0*x*y*y + 270.0*x*y;
    return result * exp_t2 / 1280.0;
  }
  void set_time(double t) { current_time = t; }
private:
  mutable double current_time = 0.0;
};

class SourceTermB : public Function<2>
{
public:
  virtual double value(const Point<2> &p, const unsigned int = 0) const override
  {
    double x = p[0], y = p[1], t = current_time;
    double exp_t2 = std::exp(t/2.0), result = 0.0;
    result += -1536.0*t*x*x*x*y*y*y + 4736.0*t*x*x*x*y*y + 70528.0*t*x*x*x*y - 75776.0*t*x*x*x;
    result += -13696.0*t*x*x*y*y*y + 23456.0*t*x*x*y*y + 647648.0*t*x*x*y - 375296.0*t*x*x;
    result += 73128.0*t*x*y*y*y - 232518.0*t*x*y*y + 188190.0*t*x*y + 83040.0*t*x;
    result += 256036.0*t*y*y*y - 436271.0*t*y*y - 1590965.0*t*y + 975600.0*t;
    result += -3072.0*x*x*x*y*y*y + 9472.0*x*x*x*y*y - 6400.0*x*x*x*y;
    result += -27392.0*x*x*y*y*y + 46912.0*x*x*y*y - 19520.0*x*x*y;
    result += -1200.0*x*y*y*y - 10380.0*x*y*y + 11580.0*x*y;
    result += 73800.0*y*y*y - 121950.0*y*y + 48150.0*y;
    return result * exp_t2 / 327680.0;
  }
  void set_time(double t) { current_time = t; }
private:
  mutable double current_time = 0.0;
};

int main(int argc, char *argv[])
{
  if (argc < 3) { std::cerr << "usage: heat_iface_dealii_transient <input.txt> <output.txt>\n"; return 1; }

  std::ifstream in(argv[1]);
  if (!in) { std::cerr << "cannot open input file\n"; return 1; }

  unsigned int side, nx, ny, degree, n_samples;
  double k, x0, x1, y0, y1, iface_x, f_src_const, dt, t_end;
  in >> side >> k >> x0 >> x1 >> y0 >> y1 >> iface_x >> f_src_const >> nx >> ny >> degree >> dt >> t_end >> n_samples;
  
  std::vector<double> sy(n_samples), sv(n_samples);
  for (unsigned int i = 0; i < n_samples; ++i) in >> sy[i] >> sv[i];
  if (!in || n_samples == 0) { std::cerr << "malformed input\n"; return 1; }
  
  const Sampled1D initial_samples(sy, sv);
  const bool iface_at_x1 = std::abs(iface_x - x1) < std::abs(iface_x - x0);
  const unsigned int iface_id = iface_at_x1 ? 1 : 0;
  const unsigned int outer_id = iface_at_x1 ? 0 : 1;
  const double s_out = iface_at_x1 ? 1.0 : -1.0;
  const bool is_subdomain_A = (x0 < 0.7);

  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(tria, {nx, ny}, Point<2>(x0, y0), Point<2>(x1, y1), true);

  const FE_Q<2> fe(degree);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const QGauss<2> quadrature(degree + 1);
  FEValues<2> fe_values(fe, quadrature, update_values | update_gradients | update_JxW_values);
  const QGauss<1> face_quadrature(degree + 2);
  FEFaceValues<2> fe_face_rhs(fe, face_quadrature, update_values | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell), dummy_vec(0);
  std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

  // Build mass and stiffness matrices with constraints included
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(dof_handler, outer_id, Functions::ConstantFunction<2>(0.0), constraints);
  const InterfaceFunction iface_fun(initial_samples);
  if (side == 0)
    VectorTools::interpolate_boundary_values(dof_handler, iface_id, iface_fun, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);

  SparseMatrix<double> mass_matrix(sparsity), stiffness_matrix(sparsity);
  mass_matrix = 0.; stiffness_matrix = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              cell_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
            }
      cell->get_dof_indices(local_dofs);
      constraints.distribute_local_to_global(cell_matrix, dummy_vec, local_dofs, mass_matrix, dummy_vec);
    }
  
  // Rebuild stiffness
  stiffness_matrix = 0.;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += k * (fe_values.shape_grad(i, q)[0] * fe_values.shape_grad(j, q)[0] +
                                      fe_values.shape_grad(i, q)[1] * fe_values.shape_grad(j, q)[1]) * fe_values.JxW(q);
      cell->get_dof_indices(local_dofs);
      constraints.distribute_local_to_global(cell_matrix, dummy_vec, local_dofs, stiffness_matrix, dummy_vec);
    }

  const double theta = 0.5;
  SparseMatrix<double> system_matrix(sparsity);
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(1.0/dt, system_matrix);
  system_matrix.add(theta, stiffness_matrix);

  Vector<double> solution(dof_handler.n_dofs()), old_solution(dof_handler.n_dofs());
  solution = 0.0; old_solution = 0.0;

  SourceTermA src_a; SourceTermB src_b;
  double t = 0.0;
  unsigned int n_steps = static_cast<unsigned int>(std::ceil(t_end / dt));
  
  std::cout << "Transient heat: side=" << side << " nx=" << nx << " ny=" << ny 
            << " dt=" << dt << " n_steps=" << n_steps << " t_end=" << t_end << std::endl;

  for (unsigned int step = 0; step < n_steps; ++step)
    {
      t = (step + 1) * dt;
      
      // Update interface BC if Dirichlet side
      if (side == 0)
        {
          constraints.clear();
          VectorTools::interpolate_boundary_values(dof_handler, outer_id, Functions::ConstantFunction<2>(0.0), constraints);
          VectorTools::interpolate_boundary_values(dof_handler, iface_id, iface_fun, constraints);
          constraints.close();
          
          // Rebuild system matrix with new constraints
          DynamicSparsityPattern dsp_new(dof_handler.n_dofs());
          DoFTools::make_sparsity_pattern(dof_handler, dsp_new, constraints, false);
          SparsityPattern sparsity_new;
          sparsity_new.copy_from(dsp_new);
          
          SparseMatrix<double> M_temp(sparsity_new), K_temp(sparsity_new);
          M_temp = 0.; K_temp = 0.;
          
          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              fe_values.reinit(cell);
              cell_matrix = 0.;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
              cell->get_dof_indices(local_dofs);
              constraints.distribute_local_to_global(cell_matrix, dummy_vec, local_dofs, M_temp, dummy_vec);
            }
          
          // Rebuild K
          K_temp = 0.;
          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              fe_values.reinit(cell);
              cell_matrix = 0.;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += k * (fe_values.shape_grad(i, q)[0] * fe_values.shape_grad(j, q)[0] +
                                              fe_values.shape_grad(i, q)[1] * fe_values.shape_grad(j, q)[1]) * fe_values.JxW(q);
              cell->get_dof_indices(local_dofs);
              constraints.distribute_local_to_global(cell_matrix, dummy_vec, local_dofs, K_temp, dummy_vec);
            }
          
          system_matrix = 0.;
          system_matrix.copy_from(M_temp);
          system_matrix.add(1.0/dt, system_matrix);
          system_matrix.add(theta, K_temp);
        }

      // Assemble RHS
      Vector<double> rhs(dof_handler.n_dofs());
      rhs = 0.0;
      
      Vector<double> M_old(dof_handler.n_dofs()), K_old(dof_handler.n_dofs());
      mass_matrix.vmult(M_old, old_solution);
      stiffness_matrix.vmult(K_old, old_solution);
      rhs.add(1.0/dt, M_old);
      rhs.add(theta, K_old);
      
      // Source term
      Vector<double> source_vec(dof_handler.n_dofs());
      source_vec = 0.0;
      
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell_rhs = 0.;
          
          if (is_subdomain_A) {
            src_a.set_time(t);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += src_a.value(fe_values.quadrature_point(q)) * fe_values.shape_value(i, q) * fe_values.JxW(q);
          } else {
            src_b.set_time(t);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += src_b.value(fe_values.quadrature_point(q)) * fe_values.shape_value(i, q) * fe_values.JxW(q);
          }

          if (side == 1)
            for (const unsigned int f : cell->face_indices())
              if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == iface_id)
                {
                  fe_face_rhs.reinit(cell, f);
                  for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                    {
                      const double g = initial_samples(fe_face_rhs.quadrature_point(q)[1]);
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        cell_rhs(i) += g * fe_face_rhs.shape_value(i, q) * fe_face_rhs.JxW(q);
                    }
                }

          cell->get_dof_indices(local_dofs);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            source_vec(local_dofs[i]) += cell_rhs(i);
        }
      
      rhs.add(1.0, source_vec);
      constraints.distribute(rhs);

      // Solve
      SolverControl control(20000, 1e-14 * rhs.l2_norm() + 1e-16);
      SolverCG<Vector<double>> solver(control);
      PreconditionSSOR<SparseMatrix<double>> precond;
      precond.initialize(system_matrix, 1.2);
      solver.solve(system_matrix, solution, rhs, precond);
      constraints.distribute(solution);
      old_solution = solution;
    }

  // Extract interface values
  const Quadrature<1> face_support(fe.get_unit_face_support_points());
  FEFaceValues<2> fe_face(fe, face_support, update_values | update_gradients | update_quadrature_points);
  std::vector<double> face_T(face_support.size());
  std::vector<Tensor<1, 2>> face_grad(face_support.size());
  std::map<double, std::array<double, 3>> iface;

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == iface_id)
        {
          fe_face.reinit(cell, f);
          fe_face.get_function_values(solution, face_T);
          fe_face.get_function_gradients(solution, face_grad);
          for (unsigned int q = 0; q < face_support.size(); ++q)
            {
              const double y = fe_face.quadrature_point(q)[1];
              const double key = std::round(y * 1e10) / 1e10;
              auto &e = iface[key];
              e[0] += face_T[q];
              e[1] += -k * s_out * face_grad[q][0];
              e[2] += 1.0;
            }
        }

  std::ofstream out(argv[2]);
  out.precision(16);
  for (const auto &[y, e] : iface)
    out << y << " " << e[0] / e[2] << " " << e[1] / e[2] << "\n";
  
  std::cout << "Final time t=" << t << ", DOFs=" << dof_handler.n_dofs() << std::endl;
  return 0;
}
