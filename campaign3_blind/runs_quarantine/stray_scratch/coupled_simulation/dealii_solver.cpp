/*
 * deal.II solver for subdomain B in the coupled heat equation problem.
 * Subdomain B: (0.625, 1.5) x (0, 1), k=4, c=1
 * This is the NEUMANN side of the coupling - receives interface flux from A,
 * applies it as Neumann BC, returns interface field.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>
#include <algorithm>

using namespace dealii;

// Source term for subdomain B
class SourceFunctionB : public Function<2>
{
public:
    SourceFunctionB(double t_val)
        : Function<2>(), t(t_val) {}
    
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
    
private:
    double t;
};

double SourceFunctionB::value(const Point<2> &p, const unsigned int) const
{
    double x = p[0];
    double y = p[1];
    double exp_t = std::exp(t / 2.0);
    
    double result = (-1536*t*x*x*x*y*y*y + 4736*t*x*x*x*y*y + 70528*t*x*x*x*y - 75776*t*x*x*x 
                    - 13696*t*x*x*y*y*y + 23456*t*x*x*y*y + 647648*t*x*x*y - 375296*t*x*x 
                    + 73128*t*x*y*y*y - 232518*t*x*y*y + 188190*t*x*y + 83040*t*x 
                    + 256036*t*y*y*y - 436271*t*y*y - 1590965*t*y + 975600*t 
                    - 3072*x*x*x*y*y*y + 9472*x*x*x*y*y - 6400*x*x*x*y 
                    - 27392*x*x*y*y*y + 46912*x*x*y*y - 19520*x*x*y 
                    - 1200*x*y*y*y - 10380*x*y*y + 11580*x*y + 73800*y*y*y - 121950*y*y + 48150*y) 
                   * exp_t / 327680.0;
    
    return result;
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc < 8)
        {
            std::cerr << "Usage: " << argv[0] << " <level> <dt> <t_end> <interface_data_file> <flux_output_file> <solution_output_file> <log_file>" << std::endl;
            return 1;
        }
        
        unsigned int level = std::stoi(argv[1]);
        double dt = std::stod(argv[2]);
        double t_end = std::stod(argv[3]);
        std::string interface_data_file = argv[4];
        std::string flux_output_file = argv[5];
        std::string solution_output_file = argv[6];
        std::string log_file = argv[7];
        
        // Mesh parameters
        double h_values[] = {1.0/8.0, 1.0/16.0, 1.0/32.0};
        double h = h_values[level - 1];
        
        // Subdomain B geometry: (0.625, 1.5) x (0, 1)
        double x_min_B = 0.625;
        double x_max_B = 1.5;
        double y_max = 1.0;
        
        // Number of cells
        unsigned int nx = static_cast<unsigned int>((x_max_B - x_min_B) / h + 0.5);
        unsigned int ny = static_cast<unsigned int>(y_max / h + 0.5);
        
        // Create triangulation
        Triangulation<2> triangulation;
        GridGenerator::subdivided_hyper_rectangle(triangulation, {nx, ny}, 
                                                   Point<2>(x_min_B, 0.0), 
                                                   Point<2>(x_max_B, y_max));
        
        // Create DoFHandler with FE_Q(1) = P1 elements
        FE_Q<2> fe(1);
        DoFHandler<2> dof_handler(triangulation);
        dof_handler.distribute_dofs(fe);
        
        unsigned int ndof = dof_handler.n_dofs();
        
        // Write log file
        std::ofstream log(log_file);
        log << "NDOF = " << ndof << std::endl;
        log << "Level = " << level << std::endl;
        log << "h = " << h << std::endl;
        log << "dt = " << dt << std::endl;
        log << "t_end = " << t_end << std::endl;
        log.close();
        
        std::cout << "deal.II solver: NDOF = " << ndof << std::endl;
        
        // Material properties for subdomain B
        double k_B = 4.0;
        double c_B = 1.0;
        
        // Identify boundary indicators using manifold IDs
        // Left boundary (interface at x=0.625): indicator 1
        // Right boundary (x=1.5): indicator 2 (Dirichlet)
        // Bottom boundary (y=0): indicator 3 (Dirichlet)
        // Top boundary (y=1): indicator 4 (Dirichlet)
        
        for (auto &cell : triangulation.active_cell_iterators())
        {
            for (unsigned int face_no = 0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
            {
                if (cell->at_boundary(face_no))
                {
                    auto face = cell->face(face_no);
                    Point<2> center = face->center();
                    
                    if (std::abs(center[0] - x_min_B) < 1e-10)
                    {
                        // Interface (left boundary of B)
                        cell->face(face_no)->set_all_manifold_ids(1);
                    }
                    else if (std::abs(center[0] - x_max_B) < 1e-10)
                    {
                        // Right boundary (Dirichlet)
                        cell->face(face_no)->set_all_manifold_ids(2);
                    }
                    else if (std::abs(center[1]) < 1e-10)
                    {
                        // Bottom boundary (Dirichlet)
                        cell->face(face_no)->set_all_manifold_ids(3);
                    }
                    else if (std::abs(center[1] - y_max) < 1e-10)
                    {
                        // Top boundary (Dirichlet)
                        cell->face(face_no)->set_all_manifold_ids(4);
                    }
                }
            }
        }
        
        // Get Dirichlet boundary dofs (right, bottom, top)
        std::set<types::boundary_id> dirichlet_boundaries = {2, 3, 4};
        
        AffineConstraints<double> constraints;
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        
        // Find Dirichlet dofs
        std::vector<unsigned int> dirichlet_dofs;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            for (unsigned int face_no = 0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
            {
                if (cell->at_boundary(face_no))
                {
                    auto face = cell->face(face_no);
                    const auto &manifold_id = face->manifold_id();
                    
                    if (dirichlet_boundaries.count(manifold_id))
                    {
                        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_face; ++v)
                        {
                            const auto vertex_index = face->vertex_index(v);
                            const auto dof_index = cell->vertex_dof_index(vertex_index, 0);
                            dirichlet_dofs.push_back(dof_index);
                        }
                    }
                }
            }
        }
        
        // Sort and unique
        std::sort(dirichlet_dofs.begin(), dirichlet_dofs.end());
        dirichlet_dofs.erase(std::unique(dirichlet_dofs.begin(), dirichlet_dofs.end()), dirichlet_dofs.end());
        
        // Add Dirichlet constraints
        for (const auto &dof : dirichlet_dofs)
        {
            constraints.add_line(dof);
        }
        
        constraints.close();
        
        // Build sparsity pattern
        DynamicSparsityPattern dsp(ndof, ndof);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
        dsp.compress();
        
        SparsityPattern sparsity;
        sparsity.copy_from(dsp);
        
        SparseMatrix<double> system_matrix;
        system_matrix.reinit(sparsity);
        
        // Quadrature
        QGauss<1> line_quadrature(2);
        QGauss<2> cell_quadrature(2);
        
        FEFaceValues<2> fe_face_values(fe, line_quadrature, update_JxW_values);
        FEValues<2> fe_values(fe, cell_quadrature, update_values | update_gradients | update_JxW_values);
        
        // Vectors
        Vector<double> solution(ndof);
        Vector<double> old_solution(ndof);
        Vector<double> rhs(ndof);
        
        // Initial condition: u = 0
        solution = 0.0;
        old_solution = 0.0;
        
        // Time stepping with Crank-Nicolson
        unsigned int n_steps = static_cast<unsigned int>(t_end / dt + 0.5);
        double current_time = 0.0;
        
        for (unsigned int step = 0; step < n_steps; ++step)
        {
            double t_n = step * dt;
            double t_np1 = (step + 1) * dt;
            double t_half = t_n + dt / 2.0;
            
            // Assemble system matrix for Crank-Nicolson
            // LHS: u*v - (dt/2)*k*grad(u)*grad(v)
            system_matrix = 0.0;
            rhs = 0.0;
            
            for (const auto &cell : triangulation.active_cell_iterators())
            {
                fe_values.reinit(cell);
                
                // Convert to std::vector for distribute_local_to_global
                std::vector<unsigned int> dof_indices_vec;
                const auto &dof_indices = cell->vertex_indices();
                for (auto idx : dof_indices)
                    dof_indices_vec.push_back(idx);
                
                const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
                
                FullMatrix<double> cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
                Vector<double> cell_rhs(n_dofs_per_cell);
                
                for (unsigned int q_point = 0; q_point < cell_quadrature.size(); ++q_point)
                {
                    double JxW = fe_values.JxW(q_point);
                    
                    // Mass matrix contribution: u*v
                    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
                            cell_matrix(i, j) += fe_values.shape_value(i, q_point) * 
                                                 fe_values.shape_value(j, q_point) * JxW;
                    
                    // Stiffness matrix contribution: -(dt/2)*k*grad(u)*grad(v)
                    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                    {
                        Tensor<1,2> grad_i = fe_values.shape_grad(i, q_point);
                        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
                        {
                            Tensor<1,2> grad_j = fe_values.shape_grad(j, q_point);
                            cell_matrix(i, j) -= (dt / 2.0) * k_B * (grad_i * grad_j) * JxW;
                        }
                    }
                    
                    // RHS: old_solution*v + (dt/2)*k*grad(old_solution)*grad(v) + dt*f*v
                    double f_val = SourceFunctionB(t_half).value(fe_values.quadrature_point(q_point));
                    
                    // Compute gradient of old_solution at quadrature point
                    Tensor<1,2> grad_old_sol;
                    for (unsigned int d = 0; d < 2; ++d)
                    {
                        for (unsigned int ii = 0; ii < n_dofs_per_cell; ++ii)
                        {
                            grad_old_sol[d] += old_solution(dof_indices_vec[ii]) * fe_values.shape_grad(ii, q_point)[d];
                        }
                    }
                    
                    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                    {
                        Tensor<1,2> grad_i = fe_values.shape_grad(i, q_point);
                        cell_rhs(i) += (old_solution(dof_indices_vec[i]) + 
                                       (dt / 2.0) * k_B * (grad_old_sol * grad_i) +
                                       dt * f_val) * 
                                      fe_values.shape_value(i, q_point) * JxW;
                    }
                }
                
                // Add to global system
                constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices_vec, system_matrix, rhs);
            }
            
            // Apply constraints
            constraints.distribute(rhs);
            
            // Solve linear system
            SolverControl solver_control(1000, 1e-12);
            SolverCG<Vector<double>> solver(solver_control);
            PreconditionIdentity preconditioner;
            
            solver.solve(system_matrix, solution, rhs, preconditioner);
            
            // Update old solution
            old_solution = solution;
            current_time = t_np1;
        }
        
        std::cout << "deal.II solver completed for level " << level << ", final time = " << t_end << std::endl;
        
        // Evaluate solution at probe points for subdomain B
        // Probe points: x = 0.625 + (i_x+0.5)*0.875/44; y = 0 + (i_y+0.5)*1/44
        std::vector<Point<2>> probe_points_B;
        std::vector<double> probe_values;
        
        for (unsigned int i_x = 0; i_x < 44; ++i_x)
        {
            for (unsigned int i_y = 0; i_y < 44; ++i_y)
            {
                double x_probe = x_min_B + (i_x + 0.5) * 0.875 / 44.0;
                double y_probe = (i_y + 0.5) * 1.0 / 44.0;
                probe_points_B.push_back(Point<2>(x_probe, y_probe));
                
                // Find cell containing this point and interpolate
                double u_val = 0.0;
                
                for (const auto &cell : triangulation.active_cell_iterators())
                {
                    const auto &bounds = cell->bounding_box();
                    if (x_probe >= bounds.lower_bound(0) && x_probe <= bounds.upper_bound(0) &&
                        y_probe >= bounds.lower_bound(1) && y_probe <= bounds.upper_bound(1))
                    {
                        // Check if point is inside triangle using barycentric coords
                        Point<2> v0 = cell->vertex(0);
                        Point<2> v1 = cell->vertex(1);
                        Point<2> v2 = cell->vertex(2);
                        
                        double denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + 
                                      (v2[0] - v1[0]) * (v0[1] - v2[1]);
                        
                        if (std::abs(denom) > 1e-12)
                        {
                            double a = ((v1[1] - v2[1]) * (x_probe - v2[0]) + 
                                       (v2[0] - v1[0]) * (y_probe - v2[1])) / denom;
                            double b = ((v2[1] - v0[1]) * (x_probe - v2[0]) + 
                                       (v0[0] - v2[0]) * (y_probe - v2[1])) / denom;
                            double c = 1.0 - a - b;
                            
                            if (a >= -1e-6 && a <= 1.0+1e-6 && 
                                b >= -1e-6 && b <= 1.0+1e-6 && 
                                c >= -1e-6 && c <= 1.0+1e-6)
                            {
                                std::vector<unsigned int> dof_indices_vec;
                                const auto &dof_indices = cell->vertex_indices();
                                for (auto idx : dof_indices)
                                    dof_indices_vec.push_back(idx);
                                
                                u_val = a * solution(dof_indices_vec[0]) + 
                                       b * solution(dof_indices_vec[1]) + 
                                       c * solution(dof_indices_vec[2]);
                                break;
                            }
                        }
                    }
                }
                
                probe_values.push_back(u_val);
            }
        }
        
        // Write solution CSV
        std::ofstream sol_file(solution_output_file);
        sol_file << "x, y, u\n";
        for (size_t i = 0; i < probe_points_B.size(); ++i)
        {
            sol_file << std::scientific << std::setprecision(16)
                     << probe_points_B[i][0] << ", " 
                     << probe_points_B[i][1] << ", " 
                     << probe_values[i] << "\n";
        }
        sol_file.close();
        
        // Compute and write interface flux
        // Interface is at x = 0.625
        // Outward normal from B at interface is (-1, 0)
        // qn = -k * grad(u) . n = -k * (-du/dx) = k * du/dx
        
        std::vector<Point<2>> interface_probes;
        std::vector<double> interface_u_values;
        std::vector<double> interface_fluxes;
        
        for (unsigned int i = 0; i < 44; ++i)
        {
            double y_probe = 0.25 + (i + 0.5) * 0.5 / 44.0;
            interface_probes.push_back(Point<2>(x_min_B, y_probe));
            
            // Find cell adjacent to interface containing this y-coordinate
            double u_val = 0.0;
            double qn = 0.0;
            
            for (const auto &cell : triangulation.active_cell_iterators())
            {
                const auto &bounds = cell->bounding_box();
                
                // Check if cell touches the interface
                if (std::abs(bounds.lower_bound(0) - x_min_B) < 1e-10)
                {
                    // Check if this cell contains the y-coordinate
                    if (y_probe >= bounds.lower_bound(1) && y_probe <= bounds.upper_bound(1))
                    {
                        Point<2> v0 = cell->vertex(0);
                        Point<2> v1 = cell->vertex(1);
                        Point<2> v2 = cell->vertex(2);
                        
                        std::vector<unsigned int> dof_indices_vec;
                        const auto &dof_indices = cell->vertex_indices();
                        for (auto idx : dof_indices)
                            dof_indices_vec.push_back(idx);
                        
                        double u0 = solution(dof_indices_vec[0]);
                        double u1 = solution(dof_indices_vec[1]);
                        double u2 = solution(dof_indices_vec[2]);
                        
                        // Gradient components
                        double denom = v0[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-v1[1]);
                        
                        if (std::abs(denom) > 1e-12)
                        {
                            double dudx = (u0*(v1[1]-v2[1]) + u1*(v2[1]-v0[1]) + u2*(v0[1]-v1[1])) / denom;
                            
                            // Outward normal from B is (-1, 0)
                            // qn = -k * grad(u) . n = -k * (-dudx) = k * dudx
                            qn = k_B * dudx;
                            
                            // Interpolate u at interface point
                            // Find vertices on interface edge
                            std::vector<std::pair<Point<2>, double> > iface_verts;
                            for (unsigned int j = 0; j < 3; ++j)
                            {
                                Point<2> vc = cell->vertex(j);
                                if (std::abs(vc[0] - x_min_B) < 1e-10)
                                {
                                    iface_verts.push_back(std::make_pair(vc, solution(dof_indices_vec[j])));
                                }
                            }
                            
                            if (iface_verts.size() == 2)
                            {
                                double dy_edge = iface_verts[1].first[1] - iface_verts[0].first[1];
                                if (std::abs(dy_edge) > 1e-12)
                                {
                                    double t_param = (y_probe - iface_verts[0].first[1]) / dy_edge;
                                    u_val = iface_verts[0].second + t_param * (iface_verts[1].second - iface_verts[0].second);
                                }
                                else
                                {
                                    u_val = iface_verts[0].second;
                                }
                            }
                            
                            break;
                        }
                    }
                }
            }
            
            interface_u_values.push_back(u_val);
            interface_fluxes.push_back(qn);
        }
        
        // Write interface CSV
        std::ofstream iface_file(flux_output_file);
        iface_file << "x, y, u, qn\n";
        for (size_t i = 0; i < interface_probes.size(); ++i)
        {
            iface_file << std::scientific << std::setprecision(16)
                       << interface_probes[i][0] << ", " 
                       << interface_probes[i][1] << ", " 
                       << interface_u_values[i] << ", " 
                       << interface_fluxes[i] << "\n";
        }
        iface_file.close();
        
        std::cout << "deal.II solver finished. Output written to " << solution_output_file << " and " << flux_output_file << std::endl;
        
        return 0;
    }
    catch (ExceptionBase &exc)
    {
        exc.print_info(std::cerr);
        std::cerr << std::endl << "Error!" << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }
}
