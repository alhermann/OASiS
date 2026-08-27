#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <cmath>
#include <iomanip>

using namespace dealii;

// Coefficient function a(x,y) = 2 + exp(-x)*sin(pi*y)
class CoefficientFunction : public Function<2>
{
public:
    CoefficientFunction() : Function<2>(2) {}
    
    virtual double operator()(const Point<2> &p) const override
    {
        return 2.0 + std::exp(-p[0]) * std::sin(numbers::PI * p[1]);
    }
};

// Source term f(x,y) - the long expression given in the problem
class SourceTerm : public Function<2>
{
public:
    SourceTerm() : Function<2>(2) {}
    
    virtual double operator()(const Point<2> &p) const override
    {
        double x = p[0];
        double y = p[1];
        double pi = numbers::PI;
        double exp_neg_x = std::exp(-x);
        double sin_pi_y = std::sin(pi * y);
        double cos_pi_y = std::cos(pi * y);
        
        // Breaking down the expression into manageable parts
        double f = 0.0;
        
        // Term by term from the given expression
        f += -3*pi*x*x*x*y*y*exp_neg_x*cos_pi_y/2;
        f += -6*x*x*x*y;
        f += -3*x*x*x*y*exp_neg_x*sin_pi_y;
        f += 3*pi*x*x*x*y*exp_neg_x*cos_pi_y;
        f += 6*x*x*x;
        f += 3*x*x*x*exp_neg_x*sin_pi_y;
        f += -pi*x*x*x*exp_neg_x*cos_pi_y;
        f += 3*x*x*y*y*y*exp_neg_x*sin_pi_y/2;
        f += -9*x*x*y*y*exp_neg_x*sin_pi_y/2;
        f += -3*pi*x*x*y*y*exp_neg_x*cos_pi_y/2;
        f += -6*x*x*y;
        f += -6*pi*x*x*y*exp_neg_x*cos_pi_y;
        f += -12*x*x;
        f += -6*x*x*exp_neg_x*sin_pi_y;
        f += 7*pi*x*x*exp_neg_x*cos_pi_y/2;
        f += -6*x*y*y*y;
        f += -2*x*y*y*y*exp_neg_x*sin_pi_y;
        f += 18*x*y*y;
        f += 15*x*y*y*exp_neg_x*sin_pi_y;
        f += 3*pi*x*y*y*exp_neg_x*cos_pi_y;
        f += -7*x*y*exp_neg_x*sin_pi_y;
        f += 3*pi*x*y*exp_neg_x*cos_pi_y;
        f += 6*x;
        f += 3*x*exp_neg_x*sin_pi_y;
        f += -5*pi*x*exp_neg_x*cos_pi_y/2;
        f += -2*y*y*y;
        f += -2*y*y*y*exp_neg_x*sin_pi_y;
        f += -12*y*y;
        f += -15*y*y*exp_neg_x*sin_pi_y/2;
        f += 14*y;
        f += 19*y*exp_neg_x*sin_pi_y/2;
        
        return f;
    }
};

int main()
{
    try
    {
        // Mesh refinement levels: N = 8, 16, 32, 64 cells per side
        const unsigned int n_cells[] = {8, 16, 32, 64};
        const unsigned int n_levels = 4;
        
        for (unsigned int level = 0; level < n_levels; ++level)
        {
            const unsigned int n = n_cells[level];
            
            Triangulation<2> triangulation;
            GridGenerator::hyper_cube(triangulation, 0, 1);
            triangulation.refine_global(level);
            
            FE_Q<2> fe(1);  // Q1 elements (bilinear)
            DoFHandler<2> dof_handler(triangulation);
            dof_handler.distribute_dofs();
            
            const unsigned int n_dofs = dof_handler.n_dofs();
            
            // Write log file
            std::ofstream logfile("run_level" + std::to_string(level+1) + ".log");
            logfile << "NDOF = " << n_dofs << std::endl;
            logfile.close();
            
            SparsityPattern sparsity_pattern(dof_handler.get_communicator());
            DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);
            sparsity_pattern.compress();
            
            SparseMatrix<double> system_matrix;
            system_matrix.reinit(sparsity_pattern);
            
            Vector<double> system_rhs(n_dofs);
            Vector<double> solution(n_dofs);
            
            // Quadrature rule of degree 3 (need at least degree 3)
            QGauss<2> quadrature(3);
            
            FEValues<2> fe_values(fe, quadrature, update_values | update_gradients | 
                                  update_JxW_values | update_quadrature_points);
            
            const unsigned int n_q_points = quadrature.size();
            const unsigned int n_components = fe.n_components();
            
            CoefficientFunction coeff_func;
            SourceTerm source_func;
            
            std::vector<double> coeff_values(n_q_points);
            std::vector<double> source_values(n_q_points);
            
            std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
            
            // Assemble system matrix and RHS
            for (auto &cell : dof_handler.active_cell_iterators())
            {
                fe_values.reinit(cell);
                
                fe_values.get_function_values(coeff_func, coeff_values);
                fe_values.get_function_values(source_func, source_values);
                
                const std::vector<Tensor<1,double> >& shape_value = fe_values.shape_values();
                const std::vector<std::vector<Tensor<1,double>> >& shape_grad = fe_values.shape_gradients();
                const std::vector<double>& JxW = fe_values.JxW_values();
                
                FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
                Vector<double> cell_vector(fe.dofs_per_cell);
                
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                    {
                        for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                        {
                            cell_matrix(i,j) += coeff_values[q_point] *
                                (shape_grad[i][q_point] * shape_grad[j][q_point]) *
                                JxW[q_point];
                        }
                        cell_vector(i) += source_values[q_point] *
                            shape_value[i][q_point] * JxW[q_point];
                    }
                }
                
                cell->get_dof_indices(local_dof_indices);
                system_matrix.add(local_dof_indices, local_dof_indices, cell_matrix);
                system_rhs.add(local_dof_indices, cell_vector);
            }
            
            // Apply Dirichlet boundary conditions (u = 0 on all boundaries)
            std::map<unsigned int, unsigned int> boundary_to_index;
            boundary_to_index[0] = 0;
            
            std::vector<unsigned int> constraint_dofs;
            DoFTools::extract_boundary_dofs(dof_handler, boundary_to_index, constraint_dofs);
            
            AffineConstraints<double> constraints;
            constraints.set_zero(constraint_dofs);
            constraints.close();
            
            constraints.distribute_to_matrix(system_matrix);
            constraints.distribute_to_vector(system_rhs, 0.0);
            
            // Solve linear system
            SolverControl solver_control(1000, 1e-12);
            SolverCG<double> solver(solver_control);
            PreconditionSSOR<double> preconditioner(system_matrix, 1.0);
            
            solver.solve(system_matrix, solution, system_rhs, preconditioner);
            
            // Generate probe points and evaluate solution
            // Probe points: x = (i_x+0.5)/44, y = (i_y+0.5)/44 for i_x, i_y = 0..43
            // Ordered with last index varying fastest (y varies fastest)
            std::ofstream csvfile("solution_level" + std::to_string(level+1) + ".csv");
            csvfile << std::setprecision(15);
            csvfile << "x, y, u" << std::endl;
            
            for (unsigned int i_x = 0; i_x < 44; ++i_x)
            {
                for (unsigned int i_y = 0; i_y < 44; ++i_y)
                {
                    double x = (i_x + 0.5) / 44.0;
                    double y = (i_y + 0.5) / 44.0;
                    
                    Point<2> probe_point(x, y);
                    
                    // Find which cell contains this point
                    typename Triangulation<2>::active_cell_iterator cell = 
                        triangulation.locate_cell(probe_point);
                    
                    if (cell != typename Triangulation<2>::end())
                    {
                        // Evaluate solution at this point
                        std::vector<double> values(1);
                        std::vector<Point<2>> points(1);
                        points[0] = probe_point;
                        
                        cell->get_fe().value_list(points, values, cell->get_dof_indices(), solution);
                        
                        csvfile << std::scientific << std::setprecision(12)
                               << x << ", " << y << ", " << values[0] << std::endl;
                    }
                    else
                    {
                        // Point not found (shouldn't happen for interior points)
                        csvfile << std::scientific << std::setprecision(12)
                               << x << ", " << y << ", " << 0.0 << std::endl;
                    }
                }
            }
            
            csvfile.close();
            
            std::cout << "Level " << (level+1) << ": N=" << n << ", NDOF=" << n_dofs 
                      << ", iterations=" << solver_control.last_step() << std::endl;
        }
        
        // Compute mesh independence metric
        // Read the two finest levels and compute max relative change
        std::vector<double> u_fine(1936), u_coarse(1936);
        
        std::ifstream fine_file("solution_level4.csv");
        std::string line;
        getline(fine_file, line); // skip header
        unsigned int idx = 0;
        while (getline(fine_file, line) && idx < 1936)
        {
            size_t comma1 = line.find(',');
            size_t comma2 = line.find(',', comma1 + 1);
            u_fine[idx++] = std::stod(line.substr(comma2 + 1));
        }
        fine_file.close();
        
        std::ifstream coarse_file("solution_level3.csv");
        getline(coarse_file, line); // skip header
        idx = 0;
        while (getline(coarse_file, line) && idx < 1936)
        {
            size_t comma1 = line.find(',');
            size_t comma2 = line.find(',', comma1 + 1);
            u_coarse[idx++] = std::stod(line.substr(comma2 + 1));
        }
        coarse_file.close();
        
        double max_rel_change = 0.0;
        for (unsigned int i = 0; i < 1936; ++i)
        {
            double rel_change = std::abs(u_fine[i] - u_coarse[i]) / 
                               (std::abs(u_coarse[i]) > 1e-15 ? std::abs(u_coarse[i]) : 1.0);
            if (rel_change > max_rel_change)
                max_rel_change = rel_change;
        }
        
        bool converged = (max_rel_change < 0.01); // 1% threshold
        
        std::ofstream result_file("RESULT.txt");
        result_file << "LEVELS = 4" << std::endl;
        result_file << "FILES = solution_level1.csv, solution_level2.csv, solution_level3.csv, solution_level4.csv" << std::endl;
        result_file << "MESH_INDEPENDENCE = " << (converged ? "CONVERGED" : "NOT_CONVERGED") << std::endl;
        result_file << std::scientific << std::setprecision(12);
        result_file << "MAX_REL_CHANGE = " << max_rel_change << std::endl;
        result_file.close();
        
        std::cout << "Max relative change between levels 3 and 4: " << max_rel_change << std::endl;
        std::cout << "Mesh independence: " << (converged ? "CONVERGED" : "NOT_CONVERGED") << std::endl;
        
    }
    catch (Exception &exc)
    {
        exc.print_info(std::cerr);
        std::ofstream result_file("RESULT.txt");
        result_file << "COULD_NOT_COMPLETE" << std::endl;
        result_file << "Error: " << exc.what() << std::endl;
        result_file.close();
        return 1;
    }
    
    return 0;
}
