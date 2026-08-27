#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace dealii;

// Source term in subdomain A
class SourceTermA : public Function<2>
{
public:
    SourceTermA() : Function<2>(2) {}
    
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override
    {
        double x = p[0];
        double y = p[1];
        return 9*x*x*x*y + 2*x*x*x + 27*x*x*y*y/4 + 129*x*x*y/16 - 319*x*x/24 
             + 9*x*y*y*y/2 + 177*x*y*y/32 - 631*x*y/24 + 323*x/32 
             + 27*y*y*y/32 - 85*y*y/12 + 673*y/96 - 37/48;
    }
};

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0] << " <level> <input_interface> <output_solution> <output_interface> <output_log> <output_flux>\n";
        return 1;
    }
    
    unsigned int level = std::stoi(argv[1]);
    std::string input_interface = argv[2];
    std::string output_solution = argv[3];
    std::string output_interface = argv[4];
    std::string output_log = argv[5];
    std::string output_flux = argv[6];
    
    // Parameters
    double domain_width = 0.625;
    double domain_height = 1.0;
    double interface_x = 5.0/8.0;
    
    // Conductivity tensor K = [[1, 1/2], [1/2, 2]]
    Tensor<2,2> K;
    K[0][0] = 1.0; K[0][1] = 0.5;
    K[1][0] = 0.5; K[1][1] = 2.0;
    
    // Create mesh
    Triangulation<2> triangulation;
    Point<2> p0(0.0, 0.0);
    Point<2> p1(domain_width, domain_height);
    GridGenerator::hyper_rectangle(triangulation, p0, p1);
    triangulation.refine_global(level);
    
    // Setup FE and DOFs
    FE_Q<2> fe(1);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    
    unsigned int n_dofs = dof_handler.n_dofs();
    
    // Read interface values
    std::vector<Point<2>> interface_points;
    std::vector<double> interface_values;
    
    std::ifstream infile(input_interface);
    std::string line;
    getline(infile, line); // Skip header
    while (getline(infile, line))
    {
        std::stringstream ss(line);
        double x, y, u;
        char comma;
        ss >> x >> comma >> y >> comma >> u;
        interface_points.push_back(Point<2>(x, y));
        interface_values.push_back(u);
    }
    infile.close();
    
    // Identify boundary DOFs
    std::vector<bool> is_dirichlet(n_dofs, false);
    std::vector<double> dirichlet_value(n_dofs, 0.0);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        
        std::vector<unsigned int> local_dofs(fe.n_dofs_per_cell());
        cell->get_dof_indices(local_dofs);
        
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
            Point<2> vertex = cell->vertex(v);
            unsigned int global_dof = local_dofs[v];
            
            // Check if on outer boundary (not interface)
            bool on_left = std::abs(vertex[0]) < 1e-12;
            bool on_bottom = std::abs(vertex[1]) < 1e-12;
            bool on_top = std::abs(vertex[1] - 1.0) < 1e-12;
            bool on_right = std::abs(vertex[0] - domain_width) < 1e-12;
            
            if (on_left || on_bottom || on_top)
            {
                is_dirichlet[global_dof] = true;
                dirichlet_value[global_dof] = 0.0;
            }
            else if (on_right && vertex[1] > 1e-10 && vertex[1] < 1.0 - 1e-10)
            {
                // Interface DOF - interpolate from interface values
                is_dirichlet[global_dof] = true;
                // Find closest interface point
                double min_dist = 1e10;
                double interp_val = 0.0;
                for (size_t k = 0; k < interface_points.size(); ++k)
                {
                    double dist = std::abs(interface_points[k][1] - vertex[1]);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        interp_val = interface_values[k];
                    }
                }
                dirichlet_value[global_dof] = interp_val;
            }
        }
    }
    
    // Assemble system using standard deal.II classes
    QGauss<2> quadrature(fe.degree + 1);
    FEValues<2> fe_values(fe, quadrature,
                          update_values | update_quadrature_points | update_JxW_values | update_gradients);
    
    SparsityPattern sparsity(n_dofs, n_dofs);
    SparseMatrix<double> system_matrix;
    Vector<double> system_rhs;
    Vector<double> solution;
    
    // Build sparsity pattern
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();
    system_matrix.reinit(sparsity);
    system_rhs.reinit(n_dofs);
    solution.reinit(n_dofs);
    
    SourceTermA source_term;
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        
        fe_values.reinit(cell);
        
        std::vector<unsigned int> local_dofs(fe.n_dofs_per_cell());
        cell->get_dof_indices(local_dofs);
        
        std::vector<double> local_rhs(fe.n_dofs_per_cell(), 0.0);
        std::vector<std::vector<double>> local_matrix(fe.n_dofs_per_cell(), 
                                                       std::vector<double>(fe.n_dofs_per_cell(), 0.0));
        
        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            double f_val = source_term.value(fe_values.quadrature_point(q));
            double JxW = fe_values.JxW(q);
            
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
                local_rhs[i] += f_val * fe_values.shape_value(i, q) * JxW;
                
                Tensor<1,2> grad_i = fe_values.shape_grad(i, q);
                
                for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                {
                    Tensor<1,2> grad_j = fe_values.shape_grad(j, q);
                    
                    // K grad phi_j . grad phi_i
                    double Kgrad_j_0 = K[0][0] * grad_j[0] + K[0][1] * grad_j[1];
                    double Kgrad_j_1 = K[1][0] * grad_j[0] + K[1][1] * grad_j[1];
                    double integrand = (Kgrad_j_0 * grad_i[0] + Kgrad_j_1 * grad_i[1]) * JxW;
                    
                    local_matrix[i][j] += integrand;
                }
            }
        }
        
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        {
            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
            {
                system_matrix.add(local_dofs[i], local_dofs[j], local_matrix[i][j]);
            }
            system_rhs(local_dofs[i]) += local_rhs[i];
        }
    }
    
    system_matrix.compress(VectorOperation::add);
    
    // Apply Dirichlet BCs
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
        if (is_dirichlet[i])
        {
            for (unsigned int j = 0; j < n_dofs; ++j)
            {
                if (i == j)
                {
                    system_matrix.set(i, j, 1.0);
                    system_rhs(i) = dirichlet_value[i];
                }
                else
                {
                    system_matrix.set(i, j, 0.0);
                }
            }
        }
    }
    
    // Solve
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    
    // Write log file
    std::ofstream logfile(output_log);
    logfile << "NDOF = " << n_dofs << "\n";
    logfile.close();
    
    // Generate probe points for subdomain A
    std::vector<Point<2>> probe_points;
    for (unsigned int i_y = 0; i_y < 44; ++i_y)
    {
        for (unsigned int i_x = 0; i_x < 44; ++i_x)
        {
            double x = 0.0 + (i_x + 0.5) * 0.625 / 44.0;
            double y = 0.0 + (i_y + 0.5) * 1.0 / 44.0;
            probe_points.push_back(Point<2>(x, y));
        }
    }
    
    // Interpolate solution at probe points
    std::ofstream solfile(output_solution);
    solfile << std::setprecision(15);
    solfile << "x,y,u\n";
    
    for (size_t i = 0; i < probe_points.size(); ++i)
    {
        Point<2> probe = probe_points[i];
        double u_val = 0.0;
        
        // Find cell containing this point
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            
            // Get cell vertices to determine bounds
            Point<2> min_pt = cell->vertex(0);
            Point<2> max_pt = cell->vertex(0);
            for (unsigned int v = 1; v < 4; ++v)
            {
                Point<2> vpt = cell->vertex(v);
                if (vpt[0] < min_pt[0]) min_pt[0] = vpt[0];
                if (vpt[1] < min_pt[1]) min_pt[1] = vpt[1];
                if (vpt[0] > max_pt[0]) max_pt[0] = vpt[0];
                if (vpt[1] > max_pt[1]) max_pt[1] = vpt[1];
            }
            
            if (probe[0] >= min_pt[0] - 1e-12 && probe[0] <= max_pt[0] + 1e-12 &&
                probe[1] >= min_pt[1] - 1e-12 && probe[1] <= max_pt[1] + 1e-12)
            {
                std::vector<unsigned int> local_dofs(fe.n_dofs_per_cell());
                cell->get_dof_indices(local_dofs);
                
                // Linear interpolation for P1 elements
                double dx = max_pt[0] - min_pt[0];
                double dy = max_pt[1] - min_pt[1];
                double xi = (probe[0] - min_pt[0]) / dx;
                double eta = (probe[1] - min_pt[1]) / dy;
                
                // Shape functions for bilinear quad
                double N0 = (1-xi)*(1-eta);
                double N1 = xi*(1-eta);
                double N2 = xi*eta;
                double N3 = (1-xi)*eta;
                
                u_val = N0 * solution(local_dofs[0]) + N1 * solution(local_dofs[1]) + 
                       N2 * solution(local_dofs[2]) + N3 * solution(local_dofs[3]);
                break;
            }
        }
        
        solfile << probe[0] << "," << probe[1] << "," << u_val << "\n";
    }
    solfile.close();
    
    // Compute interface flux and values
    std::vector<Point<2>> interface_probe_points;
    for (unsigned int i = 0; i < 44; ++i)
    {
        double y = 0.25 + (i + 0.5) * 0.5 / 44.0;
        interface_probe_points.push_back(Point<2>(interface_x, y));
    }
    
    std::ofstream ifacefile(output_interface);
    ifacefile << std::setprecision(15);
    ifacefile << "x,y,u,qn\n";
    
    std::ofstream fluxfile(output_flux);
    fluxfile << std::setprecision(15);
    fluxfile << "x,y,qn\n";
    
    for (size_t i = 0; i < interface_probe_points.size(); ++i)
    {
        Point<2> probe = interface_probe_points[i];
        double u_val = 0.0;
        Tensor<1,2> grad_u;
        grad_u[0] = 0.0; grad_u[1] = 0.0;
        
        // Find cell adjacent to interface
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            
            Point<2> center = cell->center();
            if (std::abs(center[0] - domain_width + 0.5*(domain_width/cell->level())) < 1e-10 &&
                std::abs(probe[1] - center[1]) < 0.5*domain_height/cell->level())
            {
                // Get solution at interface point
                std::vector<unsigned int> local_dofs(fe.n_dofs_per_cell());
                cell->get_dof_indices(local_dofs);
                
                // Get cell vertices to determine bounds
                Point<2> min_pt = cell->vertex(0);
                Point<2> max_pt = cell->vertex(0);
                for (unsigned int v = 1; v < 4; ++v)
                {
                    Point<2> vpt = cell->vertex(v);
                    if (vpt[0] < min_pt[0]) min_pt[0] = vpt[0];
                    if (vpt[1] < min_pt[1]) min_pt[1] = vpt[1];
                    if (vpt[0] > max_pt[0]) max_pt[0] = vpt[0];
                    if (vpt[1] > max_pt[1]) max_pt[1] = vpt[1];
                }
                
                double dx = max_pt[0] - min_pt[0];
                double dy = max_pt[1] - min_pt[1];
                double xi = (probe[0] - min_pt[0]) / dx;
                double eta = (probe[1] - min_pt[1]) / dy;
                
                double N0 = (1-xi)*(1-eta);
                double N1 = xi*(1-eta);
                double N2 = xi*eta;
                double N3 = (1-xi)*eta;
                
                u_val = N0 * solution(local_dofs[0]) + N1 * solution(local_dofs[1]) + 
                       N2 * solution(local_dofs[2]) + N3 * solution(local_dofs[3]);
                
                // Compute gradient
                // For bilinear element: dN0/dx = -(1-eta)/dx, etc.
                double dN0_dx = -(1-eta)/dx, dN0_dy = -(1-xi)/dy;
                double dN1_dx = (1-eta)/dx, dN1_dy = -xi/dy;
                double dN2_dx = eta/dx, dN2_dy = xi/dy;
                double dN3_dx = -eta/dx, dN3_dy = (1-xi)/dy;
                
                grad_u[0] = dN0_dx*solution(local_dofs[0]) + dN1_dx*solution(local_dofs[1]) +
                           dN2_dx*solution(local_dofs[2]) + dN3_dx*solution(local_dofs[3]);
                grad_u[1] = dN0_dy*solution(local_dofs[0]) + dN1_dy*solution(local_dofs[1]) +
                           dN2_dy*solution(local_dofs[2]) + dN3_dy*solution(local_dofs[3]);
                
                break;
            }
        }
        
        // Outward normal from subdomain A at x=0.625 is (1, 0)
        Tensor<1,2> n;
        n[0] = 1.0; n[1] = 0.0;
        
        // qn = -(K grad u) . n
        Tensor<1,2> Kgrad_u;
        Kgrad_u[0] = K[0][0] * grad_u[0] + K[0][1] * grad_u[1];
        Kgrad_u[1] = K[1][0] * grad_u[0] + K[1][1] * grad_u[1];
        
        double qn = -(Kgrad_u[0] * n[0] + Kgrad_u[1] * n[1]);
        
        ifacefile << probe[0] << "," << probe[1] << "," << u_val << "," << qn << "\n";
        fluxfile << probe[0] << "," << probe[1] << "," << qn << "\n";
    }
    ifacefile.close();
    fluxfile.close();
    
    return 0;
}
