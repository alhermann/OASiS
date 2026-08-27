#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/indices.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/fe_collection.h>

#include <fstream>
#include <iomanip>
#include <cmath>

using namespace dealii;

// Material parameters
const double E = 2000.0;
const double nu = 0.25;
const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); // Should be 800
const double mu = E / (2 * (1 + nu)); // Should be 800

// Source term function
class SourceTerm : public Function<3>
{
public:
    SourceTerm() : Function<3>(3) {}

    virtual void vector_value(const Point<3> &p, Vector<double> &values) const override
    {
        double x = p(0);
        double y = p(1);
        double z = p(2);

        // f_ux
        values(0) = -1600*x*x*x*y*y + 1600*x*x*x*y - 1600*x*x*x*z*z 
                   + 1600*x*x*x*z + 1600*x*x*y*y*y/3.0 + 5760*x*x*y*y*z 
                   - 1280*x*x*y*y + 40000*x*x*y*z*z - 45760*x*x*y*z 
                   + 2240*x*x*y/3.0 - 17600*x*x*z*z + 17600*x*x*z 
                   + 3200*x*y*y*y*z - 6400*x*y*y*y/3.0 - 24000*x*y*y*z*z 
                   + 4160*x*y*y*z + 9920*x*y*y - 1600*x*y*z*z 
                   + 18240*x*y*z - 23360*x*y/3.0 + 10400*x*z*z 
                   - 10400*x*z + 1600*y*y*y*z*z - 3200*y*y*y*z 
                   + 800*y*y*y + 9600*y*y*z*z - 1600*y*y*z 
                   - 4000*y*y - 12000*y*z*z + 5600*y*z 
                   + 3200*y + 1200*z*z - 1200*z;

        // f_uy
        values(1) = 6400*x*x*x*y*y + 3840*x*x*x*y*z - 8320*x*x*x*y 
                   + 19200*x*x*x*z*z - 21120*x*x*x*z + 960*x*x*x 
                   - 1600*x*x*y*y*y + 4800*x*x*y*y*z - 6000*x*x*y*y 
                   - 24000*x*x*y*z*z + 4160*x*x*y*z + 15120*x*x*y 
                   - 6000*x*x*z*z + 14320*x*x*z - 4160*x*x 
                   + 1600*x*y*y*y + 22400*x*y*y*z*z - 27200*x*y*y*z 
                   - 400*x*y*y + 1600*x*y*z*z + 14400*x*y*z 
                   - 6800*x*y - 38000*x*z*z/3.0 + 18800*x*z/3.0 
                   + 3200*x - 1600*y*y*y*z*z + 1600*y*y*y*z 
                   - 5200*y*y*z*z + 5200*y*y*z + 5200*y*z*z 
                   - 5200*y*z + 1600*z*z/3.0 - 1600*z/3.0;

        // f_uz
        values(2) = 2880*x*x*x*y*y + 25600*x*x*x*y*z - 15680*x*x*x*y 
                   + 960*x*x*x*z*z - 13760*x*x*x*z + 6400*x*x*x 
                   + 2400*x*x*y*y*y - 19200*x*x*y*y*z - 5280*x*x*y*y 
                   + 2400*x*x*y*z*z - 7200*x*x*y*z + 14880*x*x*y 
                   - 4960*x*x*z*z + 15360*x*x*z - 5200*x*x 
                   + 6400*x*y*y*y*z/3.0 - 10400*x*y*y*y/3.0 
                   + 2880*x*y*y*z*z + 13120*x*y*y*z + 4000*x*y*y 
                   - 5280*x*y*z*z - 43360*x*y*z/3.0 + 800*x*y/3.0 
                   + 4000*x*z*z - 1600*x*z - 1200*x 
                   + 800*y*y*y*z*z - 5600*y*y*y*z/3.0 + 1600*y*y*y/3.0 
                   - 4960*y*y*z*z + 4960*y*y*z + 4160*y*z*z 
                   - 9280*y*z/3.0 - 1600*y/3.0;
    }
};

// Exact solution (for verification if needed)
class ExactSolution : public Function<3>
{
public:
    ExactSolution() : Function<3>(3) {}

    virtual void vector_value(const Point<3> &p, Vector<double> &values) const override
    {
        double x = p(0);
        double y = p(1);
        double z = p(2);

        // Based on the source term, the exact solution appears to be:
        // u_x = x^2 * y^2 * z^2 * (1-x) * (1-y) * (1-z) * some polynomial
        // Let's try a simpler form that gives zero on boundary
        
        // Actually, looking at the source term structure, it seems like:
        // u = [x^2*(1-x)^2 * y^2*(1-y)^2 * z^2*(1-z)^2] * [some polynomial]
        
        // For now, we'll compute numerically without knowing exact solution
        values(0) = 0.0;
        values(1) = 0.0;
        values(2) = 0.0;
    }
};

template <int dim>
class ElasticityProblem
{
public:
    ElasticityProblem();
    
    void run();
    
private:
    void make_grid_and_dofs(unsigned int n_cells_per_side);
    void assemble_system();
    void solve();
    void evaluate_and_write_csv(unsigned int level, const std::string& filename);
    void write_log(unsigned int level, unsigned int ndof, const std::string& log_filename);
    
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    
    FE_Q<dim> fe;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> system_rhs;
    Vector<double> solution;
    
    SourceTerm source_term;
    
    unsigned int current_level;
    unsigned int n_cells_per_side;
};

template <int dim>
ElasticityProblem<dim>::ElasticityProblem()
    : dof_handler(triangulation), fe(FE_Q<dim>(1)), current_level(0), n_cells_per_side(0)
{}

template <int dim>
void ElasticityProblem<dim>::make_grid_and_dofs(unsigned int n_cells_per_side)
{
    this->n_cells_per_side = n_cells_per_side;
    
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(n_cells_per_side - 1); // Start from 1 cell, refine to get n_cells
    
    fe.distribute_dofs(dof_handler);
    
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityTools::drop_zero_entries(dsp);
    sparsity_pattern.copy_from(dsp);
}

template <int dim>
void ElasticityProblem<dim>::assemble_system()
{
    const QGauss<dim> quadrature_formula(fe.degree + 2); // At least degree 3
    const FEFaceValues<dim> fe_face_values(fe, quadrature_formula, update_values | update_quadrature_points);
    
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    
    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
    
    std::vector<Tensor<1,dim> > cell_values(n_q_points);
    std::vector<Tensor<2,dim> > cell_grads(n_q_points);
    
    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    
    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    
    for (; cell != endc; ++cell)
    {
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            
            cell_matrix = 0;
            cell_rhs = 0;
            
            // We need to integrate over the cell
            // For simplicity, use a higher order quadrature
            const QGauss<dim> q_formula(4); // Degree 4 should be enough
            
            const unsigned int n_qpts = q_formula.size();
            
            std::vector<Point<dim> > q_points(n_qpts);
            std::vector<double> q_weights(n_qpts);
            std::vector<Tensor<1,dim> > shape_grads(n_qpts, fe.n_components * fe.n_dofs_per_cell());
            std::vector<double> shape_values(n_qpts * fe.n_dofs_per_cell());
            
            for (unsigned int q = 0; q < n_qpts; ++q)
            {
                q_points[q] = cell->quadrature_point(q);
                q_weights[q] = q_formula.weight(q);
                
                for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                {
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        shape_grads[q][i*dim + d] = cell->gradient(i, q)[d];
                    }
                    shape_values[q * fe.n_dofs_per_cell() + i] = cell->value(i, q);
                }
            }
            
            // Assemble stiffness matrix
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    for (unsigned int q = 0; q < n_qpts; ++q)
                    {
                        Tensor<1,dim> grad_phi_i, grad_phi_j;
                        for (unsigned int d = 0; d < dim; ++d)
                        {
                            grad_phi_i[d] = shape_grads[q][i*dim + d];
                            grad_phi_j[d] = shape_grads[q][j*dim + d];
                        }
                        
                        // Strain tensors
                        Tensor<2,dim> eps_i, eps_j;
                        for (unsigned int a = 0; a < dim; ++a)
                        {
                            for (unsigned int b = 0; b < dim; ++b)
                            {
                                eps_i[a][b] = 0.5 * (grad_phi_i[a] * (i % dim == b) + grad_phi_i[b] * (i % dim == a));
                                eps_j[a][b] = 0.5 * (grad_phi_j[a] * (j % dim == b) + grad_phi_j[b] * (j % dim == a));
                            }
                        }
                        
                        // Stress-strain relation: sigma = 2*mu*eps + lambda*tr(eps)*I
                        double tr_eps_j = 0;
                        for (unsigned int a = 0; a < dim; ++a)
                            tr_eps_j += eps_j[a][a];
                        
                        Tensor<2,dim> sigma_j;
                        for (unsigned int a = 0; a < dim; ++a)
                        {
                            for (unsigned int b = 0; b < dim; ++b)
                            {
                                sigma_j[a][b] = 2 * mu * eps_j[a][b] + lambda * tr_eps_j * (a == b ? 1 : 0);
                            }
                        }
                        
                        // Integrand: eps_i : sigma_j
                        double integrand = 0;
                        for (unsigned int a = 0; a < dim; ++a)
                        {
                            for (unsigned int b = 0; b < dim; ++b)
                            {
                                integrand += eps_i[a][b] * sigma_j[a][b];
                            }
                        }
                        
                        cell_matrix(i, j) += integrand * q_weights[q] * cell->JxW(q);
                    }
                }
            }
            
            // Assemble RHS
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int q = 0; q < n_qpts; ++q)
                {
                    Vector<double> f_val(dim);
                    source_term.vector_value(cell->quadrature_point(q), f_val);
                    
                    for (unsigned int c = 0; c < dim; ++c)
                    {
                        cell_rhs(i) += f_val(c) * shape_values[q * fe.n_dofs_per_cell() + i] * q_weights[q] * cell->JxW(q);
                    }
                }
            }
            
            cell->add_contributions(system_matrix, cell_matrix, local_dof_indices, local_dof_indices);
            cell->add_contributions(system_rhs, cell_rhs, local_dof_indices);
        }
    }
    
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    
    // Apply boundary conditions
    std::vector<bool> constrained(dof_handler.n_dofs(), false);
    std::vector<types::boundary_id> boundary_ids;
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
        boundary_ids.push_back(static_cast<types::boundary_id>(i));
    
    DoFTools::extract_boundary_dofs(dof_handler, boundary_ids, constrained);
    
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
        if (constrained[i])
        {
            for (unsigned int j = 0; j < dof_handler.n_dofs(); ++j)
                system_matrix(i, j) = 0;
            system_matrix(i, i) = 1;
            system_rhs(i) = 0;
        }
    }
}

template <int dim>
void ElasticityProblem<dim>::solve()
{
    SolverControl solver_control(dof_handler.n_dofs() * 100, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);
    
    PreconditionSSOR<SparseMatrix<double>> preconditioner(system_matrix, 1.2);
    
    solution = 0;
    cg.vmult(solution, system_matrix, system_rhs);
    cg.solve(system_matrix, solution, preconditioner, system_rhs);
}

template <int dim>
void ElasticityProblem<dim>::evaluate_and_write_csv(unsigned int level, const std::string& filename)
{
    // Probe points: x = (i_x+0.5)/21, y = (i_y+0.5)/21, z = (i_z+0.5)/21
    // for i_x, i_y, i_z = 0, ..., 20
    // ordered with last index varying fastest
    
    std::ofstream out(filename);
    out << std::setprecision(15);
    out << "x,y,z,ux,uy,uz\n";
    
    const unsigned int n_probe = 21;
    
    for (unsigned int ix = 0; ix < n_probe; ++ix)
    {
        for (unsigned int iy = 0; iy < n_probe; ++iy)
        {
            for (unsigned int iz = 0; iz < n_probe; ++iz)
            {
                double x = (ix + 0.5) / 21.0;
                double y = (iy + 0.5) / 21.0;
                double z = (iz + 0.5) / 21.0;
                
                Point<3> p(x, y, z);
                
                // Find which cell contains this point
                typename Triangulation<dim>::active_cell_iterator cell = 
                    triangulation.find_cell(p);
                
                if (cell->at_vertex(0) && !cell->is_locally_owned())
                {
                    // This shouldn't happen in serial
                    out << x << "," << y << "," << z << ",0,0,0\n";
                    continue;
                }
                
                // Interpolate at this point
                std::vector<unsigned int> local_dof_indices(fe.n_dofs_per_cell());
                cell->get_dof_indices(local_dof_indices);
                
                Vector<double> u_val(3, 0.0);
                
                for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                {
                    double phi_val = cell->value(i, p);
                    for (unsigned int c = 0; c < 3; ++c)
                    {
                        u_val[c] += solution(local_dof_indices[i]) * phi_val;
                    }
                }
                
                out << std::scientific << std::setprecision(12)
                    << x << "," << y << "," << z << ","
                    << u_val[0] << "," << u_val[1] << "," << u_val[2] << "\n";
            }
        }
    }
    
    out.close();
}

template <int dim>
void ElasticityProblem<dim>::write_log(unsigned int level, unsigned int ndof, const std::string& log_filename)
{
    std::ofstream out(log_filename);
    out << "NDOF = " << ndof << "\n";
    out.close();
}

template <int dim>
void ElasticityProblem<dim>::run()
{
    const unsigned int levels[] = {4, 8, 16};
    const unsigned int n_levels = 3;
    
    for (unsigned int lvl = 0; lvl < n_levels; ++lvl)
    {
        current_level = lvl + 1;
        unsigned int n_cells = levels[lvl];
        
        std::cout << "Level " << current_level << ": N = " << n_cells << " cells per side" << std::endl;
        
        make_grid_and_dofs(n_cells);
        
        unsigned int ndof = dof_handler.n_dofs();
        std::cout << "Number of DOFs: " << ndof << std::endl;
        
        assemble_system();
        solve();
        
        // Write log file
        std::string log_filename = "run_level" + std::to_string(current_level) + ".log";
        write_log(current_level, ndof, log_filename);
        
        // Write CSV file
        std::string csv_filename = "solution_level" + std::to_string(current_level) + ".csv";
        evaluate_and_write_csv(current_level, csv_filename);
        
        std::cout << "Wrote " << csv_filename << std::endl;
    }
}

int main()
{
    try
    {
        ElasticityProblem<3> elasticity_problem;
        elasticity_problem.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }
    
    return 0;
}
