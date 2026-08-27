#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;

// Source term for subdomain B
class SourceFunctionB : public Function<2>
{
public:
    SourceFunctionB() : Function<2>(2) {}
    
    virtual void vector_value(const Point<2> &p, Vector<double> &values) const override
    {
        double x = p(0);
        double y = p(1);
        
        // f_x
        values(0) = (-101*x*x*y*y*y/1470 - 202*x*x*y*y/1225 + 1111*x*x*y/4900 - 101*x*x/2450 
                     + 11869*x*y*y*y/18375 + 47476*x*y*y/30625 - 130559*x*y/61250 + 11869*x/30625 
                     - 101*y*y*y*y*y/6125 - 404*y*y*y*y/6125 - 9591*y*y*y/39200 - 274761*y*y/245000 
                     + 2755731*y/1960000 - 250521/980000);
        
        // f_y
        values(1) = (33183*x*x*y*y/24500 + 66366*x*x*y/30625 - 365013*x*x/245000 
                     - 101*x*y*y*y*y/2100 - 404*x*y*y*y/2625 - 17141*x*y*y/9800 - 104794*x*y/30625 
                     + 1113849*x/490000 + 499*y*y*y*y/3675 + 7984*y*y*y/18375 - 41347*y*y/49000 
                     + 2509*y/6125 - 5643/98000);
    }
};

int main()
{
    try
    {
        std::cout << "deal.II solver for subdomain B" << std::endl;
        
        // Create triangulation for subdomain B: (0.625, 1.5) x (0, 1)
        Triangulation<2> triangulation;
        GridGenerator::subdivided_hyper_rectangle(
            triangulation, 
            8, 8,  // nx, ny
            Point<2>(0.625, 0), 
            Point<2>(1.5, 1.0)
        );
        
        // Set up DoF handler
        FE_Q<VectorizedArray<double>, 2> fe(1);
        DoFHandler<2> dof_handler(triangulation);
        dof_handler.distribute_dofs(fe);
        
        const unsigned int n_dofs = dof_handler.n_dofs();
        std::cout << "NDOF = " << n_dofs << std::endl;
        
        // Sparsity pattern
        SparsityPattern sparsity_pattern;
        DynamicSparsityPattern dsp(n_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        
        // Assemble system
        SparseMatrix<double> system_matrix;
        system_matrix.initialize(sparsity_pattern);
        Vector<double> system_rhs(n_dofs);
        
        QGauss<2> quadrature_formula(2);
        FEValues<2> fe_values(fe, quadrature_formula, 
                              update_values | update_quadrature_points | 
                              update_JxW_values | update_gradients);
        
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int dim = 2;
        
        Vector<double> cell_dof_values(dofs_per_cell);
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        
        // Material constants for subdomain B
        const double lambda_B = 500.0;
        const double mu_B = 1250.0;
        
        SourceFunctionB source_func;
        
        // Cell loop
        typename DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active(),
                                                     endc = dof_handler.end();
        for (; cell != endc; ++cell)
        {
            fe_values.reinit(cell);
            
            cell_matrix = 0;
            cell_rhs = 0;
            
            for (unsigned int q_point = 0; q_point < quadrature_formula.size(); ++q_point)
            {
                double JxW = fe_values.JxW(q_point);
                
                for (unsigned int v = 0; v < dofs_per_cell; ++v)
                {
                    Tensor<1,dim> grad_v = fe_values.shape_grad(v, q_point);
                    
                    for (unsigned int u = 0; u < dofs_per_cell; ++u)
                    {
                        Tensor<1,dim> grad_u = fe_values.shape_grad(u, q_point);
                        
                        // Strain tensors
                        Tensor<2,dim> eps_v(dim, dim);
                        Tensor<2,dim> eps_u(dim, dim);
                        for (unsigned int i = 0; i < dim; ++i)
                            for (unsigned int j = 0; j < dim; ++j)
                            {
                                eps_v(i,j) = 0.5 * (grad_v[i] * (i==j ? 1.0 : 0.0) + grad_v[j] * (i==j ? 1.0 : 0.0));
                                eps_u(i,j) = 0.5 * (grad_u[i] * (i==j ? 1.0 : 0.0) + grad_u[j] * (i==j ? 1.0 : 0.0));
                            }
                        
                        // Actually compute properly using Voigt notation
                        double eps_v_xx = grad_v[0];
                        double eps_v_yy = grad_v[1];
                        double eps_v_xy = 0.5 * (grad_v[1] + grad_v[0]);
                        
                        double eps_u_xx = grad_u[0];
                        double eps_u_yy = grad_u[1];
                        double eps_u_xy = 0.5 * (grad_u[1] + grad_u[0]);
                        
                        // Stress-strain relation (plane strain)
                        double sigma_v_xx = (lambda_B + 2*mu_B) * eps_v_xx + lambda_B * eps_v_yy;
                        double sigma_v_yy = lambda_B * eps_v_xx + (lambda_B + 2*mu_B) * eps_v_yy;
                        double sigma_v_xy = 2*mu_B * eps_v_xy;
                        
                        cell_matrix(v,u) += (sigma_v_xx * eps_u_xx + 
                                            sigma_v_yy * eps_u_yy + 
                                            2*sigma_v_xy * eps_u_xy) * JxW;
                    }
                    
                    // RHS
                    Tensor<1,dim> source_vec = source_func.value(fe_values.quadrature_point(q_point));
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        cell_rhs(v) -= source_vec[d] * fe_values.shape_value(v,q_point) * JxW;
                    }
                }
            }
            
            std::vector<unsigned int> local_dof_indices(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);
            system_matrix.add(local_dof_indices, cell_matrix);
            system_rhs.add(local_dof_indices, cell_rhs);
        }
        
        // Apply boundary conditions (zero on right, top, bottom)
        AffineConstraints<double> constraints;
        
        // Right boundary (x = 1.5)
        std::set<types::global_dof_index> zero_dofs_right;
        DoFTools::extract_boundary_dofs(dof_handler, 
                                         SubscriptorList({1}),  
                                         zero_dofs_right);
        for (auto dof : zero_dofs_right)
            constraints.add_line(dof);
        
        // Bottom boundary (y = 0)
        std::set<types::global_dof_index> zero_dofs_bottom;
        DoFTools::extract_boundary_dofs(dof_handler,
                                         SubscriptorList({0}),  
                                         zero_dofs_bottom);
        for (auto dof : zero_dofs_bottom)
            constraints.add_line(dof);
        
        // Top boundary (y = 1)
        std::set<types::global_dof_index> zero_dofs_top;
        DoFTools::extract_boundary_dofs(dof_handler,
                                         SubscriptorList({2}),  
                                         zero_dofs_top);
        for (auto dof : zero_dofs_top)
            constraints.add_line(dof);
        
        constraints.close();
        
        constraints.distribute_to_matrix(system_matrix);
        constraints.distribute_to_vector(system_rhs, 0.0);
        
        // Solve
        Vector<double> solution(n_dofs);
        SolverControl solver_control(1000, 1e-12);
        SolverCG<double> solver(solver_control);
        PreconditionSSOR<double> preconditioner;
        preconditioner.set_relaxation_factor(1.2);
        
        solver.solve(system_matrix, solution, system_rhs, preconditioner);
        
        std::cout << "Solved with " << solver_control.last_step() << " iterations" << std::endl;
        
        // Write NDOF to log
        std::ofstream log_file("run_level1_B.log");
        log_file << "NDOF = " << n_dofs << "\n";
        log_file.close();
        
        return 0;
    }
    catch (ExceptionHandler::exc_exc_bad_input &)
    {
        std::cerr << "Bad input!" << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }
}
