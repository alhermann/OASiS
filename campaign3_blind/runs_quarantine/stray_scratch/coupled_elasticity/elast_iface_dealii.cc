// ---------------------------------------------------------------
// Elasticity interface solver for OASiS couple driver
// ---------------------------------------------------------------
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace dealii;

template<int dim>
class ElasticityProblem
{
public:
  ElasticityProblem(int side_flag,
                    double lambda, double mu,
                    double x0, double x1, double y0, double y1,
                    double iface_x,
                    const std::vector<std::pair<double,double>>& ux_samples,
                    const std::vector<std::pair<double,double>>& uy_samples,
                    const std::vector<std::pair<double,double>>& tx_samples,
                    const std::vector<std::pair<double,double>>& ty_samples,
                    int nx, int ny);

  void solve();
  void write_interface_output(const std::string& filename);
  unsigned int get_n_dofs() const { return dof_handler.n_dofs(); }

private:
  int side;
  double lambda, mu;
  double X0, X1, Y0, Y1, IFACE_X;
  std::vector<std::pair<double,double>> ux_samples, uy_samples;
  std::vector<std::pair<double,double>> tx_samples, ty_samples;
  
  Triangulation<dim>        tria;
  DoFHandler<dim>           dof_handler;
  FESystem<dim>             fe;
  AffineConstraints<double> constraints;
  SparseMatrix<double>      system_matrix;
  Vector<double>            solution, system_rhs;
  
  QGauss<dim>               quadrature_formula;
  QGauss<dim-1>             face_quadrature_formula;
  FEValues<dim>             fe_values;
  FEFaceValues<dim>         fe_face_values;
  std::vector<types::global_dof_index> local_dof_indices;
  FullMatrix<double>        cell_matrix;
  Vector<double>            cell_rhs;
  
  void fx_function(const Point<dim>& p, double& value) const;
  void fy_function(const Point<dim>& p, double& value) const;
  
  double interpolate_ux(double y) const;
  double interpolate_uy(double y) const;
  double interpolate_tx(double y) const;
  double interpolate_ty(double y) const;
};

template<int dim>
ElasticityProblem<dim>::ElasticityProblem(int side_flag,
                                           double lam, double mue,
                                           double x0, double x1, double y0, double y1,
                                           double iface_x,
                                           const std::vector<std::pair<double,double>>& ux_samp,
                                           const std::vector<std::pair<double,double>>& uy_samp,
                                           const std::vector<std::pair<double,double>>& tx_samp,
                                           const std::vector<std::pair<double,double>>& ty_samp,
                                           int nx, int ny)
  : side(side_flag), lambda(lam), mu(mue),
    X0(x0), X1(x1), Y0(y0), Y1(y1), IFACE_X(iface_x),
    ux_samples(ux_samp), uy_samples(uy_samp),
    tx_samples(tx_samp), ty_samples(ty_samp)
{
  GridGenerator::subdivided_hyper_rectangle(tria, 
                                             std::vector<unsigned int>{static_cast<unsigned int>(nx), 
                                                                       static_cast<unsigned int>(ny)},
                                             Point<dim>(X0, Y0),
                                             Point<dim>(X1, Y1),
                                             true);
  
  fe = FESystem<dim>(FE_Q<dim>(1), dim);
  dof_handler.distribute_dofs(fe);
  
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  
  types::boundary_id outer_x_face = (std::abs(IFACE_X - X1) < 1e-12) ? 0 : 1;
  
  Functions::ZeroFunction<dim> zero_bc(2);
  VectorTools::interpolate_boundary_values(dof_handler, outer_x_face, zero_bc, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 2, zero_bc, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 3, zero_bc, constraints);
  
  if (side == 0) {
    class InterfaceDisplacement : public Function<dim>
    {
    public:
      InterfaceDisplacement(const std::vector<std::pair<double,double>>& ux_s,
                           const std::vector<std::pair<double,double>>& uy_s)
        : Function<dim>(2), ux_s(ux_s), uy_s(uy_s) {}
      
      virtual double value(const Point<dim>& p, unsigned int component) const override
      {
        double y = p[1];
        if (component == 0) {
          for (size_t i = 1; i < ux_s.size(); ++i) {
            if (y >= ux_s[i-1].first && y <= ux_s[i].first) {
              double t = (y - ux_s[i-1].first) / (ux_s[i].first - ux_s[i-1].first);
              return ux_s[i-1].second + t * (ux_s[i].second - ux_s[i-1].second);
            }
          }
          return ux_s.empty() ? 0.0 : ux_s.back().second;
        } else {
          for (size_t i = 1; i < uy_s.size(); ++i) {
            if (y >= uy_s[i-1].first && y <= uy_s[i].first) {
              double t = (y - uy_s[i-1].first) / (uy_s[i].first - uy_s[i-1].first);
              return uy_s[i-1].second + t * (uy_s[i].second - uy_s[i-1].second);
            }
          }
          return uy_s.empty() ? 0.0 : uy_s.back().second;
        }
      }
    private:
      const std::vector<std::pair<double,double>>& ux_s, &uy_s;
    };
    
    InterfaceDisplacement iface_bc(ux_samples, uy_samples);
    types::boundary_id iface_face_id = (std::abs(IFACE_X - X1) < 1e-12) ? 1 : 0;
    VectorTools::interpolate_boundary_values(dof_handler, iface_face_id, iface_bc, constraints);
  }
  
  constraints.close();
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  system_matrix.reinit(sparsity);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  
  quadrature_formula = QGauss<dim>(fe.degree + 1);
  face_quadrature_formula = QGauss<dim-1>(fe.degree + 1);
  
  fe_values.init(fe, quadrature_formula,
                 update_values | update_gradients | update_JxW_values |
                 update_quadrature_points);
  fe_face_values.init(fe, face_quadrature_formula,
                      update_values | update_gradients | update_JxW_values |
                      update_quadrature_points | update_normal_vectors);
                      
  local_dof_indices.resize(fe.n_dofs_per_cell());
  cell_matrix.reinit(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  cell_rhs.reinit(fe.n_dofs_per_cell());
}

template<int dim>
void ElasticityProblem<dim>::fx_function(const Point<dim>& p, double& value) const
{
  double x = p[0], y = p[1];
  if (X0 < 0.7) {
    value = (x*x*y*y*y)/50.0 + (6.0*x*x*y*y)/125.0 - (33.0*x*x*y)/500.0 
          + (3.0*x*x)/250.0 - (x*y*y*y)/50.0 - (6.0*x*y*y)/125.0 
          + (33.0*x*y)/500.0 - (3.0*x)/250.0 + (y*y*y*y*y)/125.0 
          + (4.0*y*y*y*y)/125.0 - (89.0*y*y*y)/500.0 - (21.0*y*y)/125.0 
          + (297.0*y)/1000.0 - 27.0/500.0;
  } else {
    value = (-101.0*x*x*y*y*y)/1470.0 - (202.0*x*x*y*y)/1225.0 
          + (1111.0*x*x*y)/4900.0 - (101.0*x*x)/2450.0 
          + (11869.0*x*y*y*y)/18375.0 + (47476.0*x*y*y)/30625.0 
          - (130559.0*x*y)/61250.0 + (11869.0*x)/30625.0 
          - (101.0*y*y*y*y*y)/6125.0 - (404.0*y*y*y*y)/6125.0 
          - (9591.0*y*y*y)/39200.0 - (274761.0*y*y)/245000.0 
          + (2755731.0*y)/1960000.0 - 250521.0/980000.0;
  }
}

template<int dim>
void ElasticityProblem<dim>::fy_function(const Point<dim>& p, double& value) const
{
  double x = p[0], y = p[1];
  if (X0 < 0.7) {
    value = (-3.0*x*x*y*y)/100.0 - (6.0*x*x*y)/125.0 + (33.0*x*x)/1000.0 
          + (3.0*x*y*y*y*y)/100.0 + (12.0*x*y*y*y)/125.0 
          - (279.0*x*y*y)/500.0 - (63.0*x*y)/125.0 + (99.0*x)/250.0 
          - (y*y*y*y)/200.0 - (2.0*y*y*y)/125.0 + (33.0*y*y)/1000.0 
          - (3.0*y)/250.0;
  } else {
    value = (33183.0*x*x*y*y)/24500.0 + (66366.0*x*x*y)/30625.0 
          - (365013.0*x*x)/245000.0 - (101.0*x*y*y*y*y)/2100.0 
          - (404.0*x*y*y*y)/2625.0 - (17141.0*x*y*y)/9800.0 
          - (104794.0*x*y)/30625.0 + (1113849.0*x)/490000.0 
          + (499.0*y*y*y*y)/3675.0 + (7984.0*y*y*y)/18375.0 
          - (41347.0*y*y)/49000.0 + (2509.0*y)/6125.0 - 5643.0/98000.0;
  }
}

template<int dim>
double ElasticityProblem<dim>::interpolate_ux(double y) const
{
  for (size_t i = 1; i < ux_samples.size(); ++i) {
    if (y >= ux_samples[i-1].first && y <= ux_samples[i].first) {
      double t = (y - ux_samples[i-1].first) / (ux_samples[i].first - ux_samples[i-1].first);
      return ux_samples[i-1].second + t * (ux_samples[i].second - ux_samples[i-1].second);
    }
  }
  return ux_samples.empty() ? 0.0 : ux_samples.back().second;
}

template<int dim>
double ElasticityProblem<dim>::interpolate_uy(double y) const
{
  for (size_t i = 1; i < uy_samples.size(); ++i) {
    if (y >= uy_samples[i-1].first && y <= uy_samples[i].first) {
      double t = (y - uy_samples[i-1].first) / (uy_samples[i].first - uy_samples[i-1].first);
      return uy_samples[i-1].second + t * (uy_samples[i].second - uy_samples[i-1].second);
    }
  }
  return uy_samples.empty() ? 0.0 : uy_samples.back().second;
}

template<int dim>
double ElasticityProblem<dim>::interpolate_tx(double y) const
{
  for (size_t i = 1; i < tx_samples.size(); ++i) {
    if (y >= tx_samples[i-1].first && y <= tx_samples[i].first) {
      double t = (y - tx_samples[i-1].first) / (tx_samples[i].first - tx_samples[i-1].first);
      return tx_samples[i-1].second + t * (tx_samples[i].second - tx_samples[i-1].second);
    }
  }
  return tx_samples.empty() ? 0.0 : tx_samples.back().second;
}

template<int dim>
double ElasticityProblem<dim>::interpolate_ty(double y) const
{
  for (size_t i = 1; i < ty_samples.size(); ++i) {
    if (y >= ty_samples[i-1].first && y <= ty_samples[i].first) {
      double t = (y - ty_samples[i-1].first) / (ty_samples[i].first - ty_samples[i-1].first);
      return ty_samples[i-1].second + t * (ty_samples[i].second - ty_samples[i-1].second);
    }
  }
  return ty_samples.empty() ? 0.0 : ty_samples.back().second;
}

template<int dim>
void ElasticityProblem<dim>::solve()
{
  system_matrix = 0;
  system_rhs = 0;
  
  for (auto cell : dof_handler.active_cell_iterators())
  {
    cell_matrix = 0;
    cell_rhs = 0;
    
    fe_values.reinit(cell);
    
    const unsigned int n_q_points = quadrature_formula.size();
    
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      double JxW = fe_values.JxW(q);
      Point<dim> qp = fe_values.quadrature_point(q);
      
      double fx_val, fy_val;
      fx_function(qp, fx_val);
      fy_function(qp, fy_val);
      
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
      {
        Tensor<1,dim> grad_phi_i = fe_values.shape_grad(i, q);
        unsigned int comp_i = i % dim;
        
        double f_val = (comp_i == 0) ? fx_val : fy_val;
        cell_rhs(i) += f_val * fe_values.shape_value(i, q) * JxW;
        
        SymmetricTensor<2,dim> strain_i;
        for (int a = 0; a < dim; ++a)
          for (int b = 0; b < dim; ++b)
            strain_i[a][b] = 0.5 * (grad_phi_i[a]*(b==static_cast<int>(comp_i)?1:0) + grad_phi_i[b]*(a==static_cast<int>(comp_i)?1:0));
        
        double div_i = trace(strain_i);
        
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
        {
          Tensor<1,dim> grad_phi_j = fe_values.shape_grad(j, q);
          unsigned int comp_j = j % dim;
          
          SymmetricTensor<2,dim> strain_j;
          for (int a = 0; a < dim; ++a)
            for (int b = 0; b < dim; ++b)
              strain_j[a][b] = 0.5 * (grad_phi_j[a]*(b==static_cast<int>(comp_j)?1:0) + grad_phi_j[b]*(a==static_cast<int>(comp_j)?1:0));
          
          double contrib = 0.0;
          for (int a = 0; a < dim; ++a)
            for (int b = 0; b < dim; ++b)
              contrib += (2.0*mu*strain_i[a][b] + lambda*div_i*(a==b?1.0:0.0)) * strain_j[a][b];
          
          cell_matrix(i, j) += contrib * JxW;
        }
      }
    }
    
    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                           local_dof_indices, system_matrix, system_rhs);
  }
  
  if (side == 1) {
    types::boundary_id iface_face_id = (std::abs(IFACE_X - X1) < 1e-12) ? 1 : 0;
    
    for (auto cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int face_idx : cell->face_indices())
      {
        const typename Triangulation<dim>::face_iterator& face = cell->face(face_idx);
        if (!face->at_boundary()) continue;
        
        types::boundary_id face_id = face->boundary_indicator();
        if (face_id != iface_face_id) continue;
        
        fe_face_values.reinit(cell, face_idx);
        
        std::vector<types::global_dof_index> local_face_dof_indices(face->n_dofs_per_entity());
        face->get_dof_indices(local_face_dof_indices);
        
        Vector<double> cell_face_rhs(fe.n_dofs_per_cell());
        cell_face_rhs = 0;
        
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        {
          for (unsigned int q = 0; q < face_quadrature_formula.size(); ++q)
          {
            unsigned int comp_i = i % dim;
            double y = fe_face_values.quadrature_point(q)[1];
            double t_q = (comp_i == 0) ? interpolate_tx(y) : interpolate_ty(y);
            cell_face_rhs(i) += t_q * fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);
          }
        }
        
        constraints.distribute_local_to_global(cell_face_rhs, local_face_dof_indices, system_rhs);
      }
    }
  }
  
  constraints.distribute(system_rhs);
  
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix);
  
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  
  constraints.distribute(solution);
  
  std::cout << "dofs=" << dof_handler.n_dofs() 
            << " cg_steps=" << solver_control.last_step()
            << " max|u|=" << solution.linfty_norm() << std::endl;
}

template<int dim>
void ElasticityProblem<dim>::write_interface_output(const std::string& filename)
{
  std::ofstream out(filename);
  
  types::boundary_id iface_face_id = (std::abs(IFACE_X - X1) < 1e-12) ? 1 : 0;
  
  std::vector<double> y_coords;
  std::vector<std::vector<double>> ux_at_y, uy_at_y, tx_at_y, ty_at_y;
  
  for (auto cell : dof_handler.active_cell_iterators())
  {
    for (unsigned int face_idx : cell->face_indices())
    {
      const typename Triangulation<dim>::face_iterator& face = cell->face(face_idx);
      if (!face->at_boundary()) continue;
      
      types::boundary_id face_id = face->boundary_indicator();
      if (face_id != iface_face_id) continue;
      
      fe_face_values.reinit(cell, face_idx);
      
      const Tensor<1,dim> normal = fe_face_values.normal_vector(0);
      
      for (unsigned int q = 0; q < face_quadrature_formula.size(); ++q)
      {
        double y = fe_face_values.quadrature_point(q)[1];
        
        Tensor<2,dim> grad_u;
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j) {
            grad_u[i][j] = 0.0;
            for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k) {
              unsigned int comp_k = k % dim;
              if (comp_k == static_cast<unsigned int>(i))
                grad_u[i][j] += fe_face_values.shape_grad(k, q)[j] * solution(local_dof_indices[k]);
            }
          }
        
        double ux = 0.0, uy = 0.0;
        for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k) {
          unsigned int comp_k = k % dim;
          if (comp_k == 0)
            ux += fe_face_values.shape_value(k, q) * solution(local_dof_indices[k]);
          else
            uy += fe_face_values.shape_value(k, q) * solution(local_dof_indices[k]);
        }
        
        SymmetricTensor<2,dim> strain;
        for (int a = 0; a < dim; ++a)
          for (int b = 0; b < dim; ++b)
            strain[a][b] = 0.5 * (grad_u[a][b] + grad_u[b][a]);
        
        double div_u = trace(strain);
        
        SymmetricTensor<2,dim> stress;
        for (int a = 0; a < dim; ++a)
          for (int b = 0; b < dim; ++b)
            stress[a][b] = 2.0*mu*strain[a][b] + lambda*div_u*(a==b?1.0:0.0);
        
        Tensor<1,dim> traction;
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j)
            traction[i] += stress[i][j] * normal[j];
        
        bool found = false;
        for (size_t i = 0; i < y_coords.size(); ++i) {
          if (std::abs(y_coords[i] - y) < 1e-10) {
            ux_at_y[i].push_back(ux);
            uy_at_y[i].push_back(uy);
            tx_at_y[i].push_back(-traction[0]);
            ty_at_y[i].push_back(-traction[1]);
            found = true;
            break;
          }
        }
        if (!found) {
          y_coords.push_back(y);
          ux_at_y.push_back({ux});
          uy_at_y.push_back({uy});
          tx_at_y.push_back({-traction[0]});
          ty_at_y.push_back({-traction[1]});
        }
      }
    }
  }
  
  std::sort(y_coords.begin(), y_coords.end());
  for (size_t i = 0; i < y_coords.size(); ++i) {
    double ux_avg = 0.0, uy_avg = 0.0, tx_avg = 0.0, ty_avg = 0.0;
    for (double v : ux_at_y[i]) ux_avg += v;
    for (double v : uy_at_y[i]) uy_avg += v;
    for (double v : tx_at_y[i]) tx_avg += v;
    for (double v : ty_at_y[i]) ty_avg += v;
    ux_avg /= ux_at_y[i].size();
    uy_avg /= uy_at_y[i].size();
    tx_avg /= tx_at_y[i].size();
    ty_avg /= ty_at_y[i].size();
    
    out << y_coords[i] << " " << ux_avg << " " << uy_avg << " " << tx_avg << " " << ty_avg << std::endl;
  }
  
  out.close();
}

template class ElasticityProblem<2>;

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " input_file output_file" << std::endl;
    return 1;
  }
  
  std::ifstream infile(argv[1]);
  if (!infile.is_open()) {
    std::cerr << "Cannot open input file: " << argv[1] << std::endl;
    return 1;
  }
  
  int side;
  double K, X0, X1, Y0, Y1, IFACE_X;
  int NX, NY, DEGREE;
  
  infile >> side >> K >> X0 >> X1 >> Y0 >> Y1 >> IFACE_X >> NX >> NY >> DEGREE;
  
  double lambda, mu;
  if (X0 < 0.7) {
    double E = 2000.0/3.0;
    double nu = 1.0/3.0;
    mu = E / (2.0 * (1.0 + nu));
    lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  } else {
    double E = 20000.0/7.0;
    double nu = 1.0/7.0;
    mu = E / (2.0 * (1.0 + nu));
    lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  }
  
  std::vector<std::pair<double,double>> ux_samples, uy_samples, tx_samples, ty_samples;
  
  int n;
  infile >> n;
  for (int i = 0; i < n; ++i) {
    double y, v;
    infile >> y >> v;
    ux_samples.push_back({y, v});
  }
  
  infile >> n;
  for (int i = 0; i < n; ++i) {
    double y, v;
    infile >> y >> v;
    uy_samples.push_back({y, v});
  }
  
  infile >> n;
  for (int i = 0; i < n; ++i) {
    double y, v;
    infile >> y >> v;
    tx_samples.push_back({y, v});
  }
  
  infile >> n;
  for (int i = 0; i < n; ++i) {
    double y, v;
    infile >> y >> v;
    ty_samples.push_back({y, v});
  }
  
  infile.close();
  
  ElasticityProblem<2> problem(side, lambda, mu, X0, X1, Y0, Y1, IFACE_X,
                               ux_samples, uy_samples, tx_samples, ty_samples,
                               NX, NY);
  
  problem.solve();
  problem.write_interface_output(argv[2]);
  
  return 0;
}
