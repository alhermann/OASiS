/* ---------------------------------------------------------------------
 * Subdomain B solver using deal.II
 * Solves linear elasticity on (0.625, 1.5) x (0, 1) with Neumann BC on interface
 * --------------------------------------------------------------------- */

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace dealii;

template<int dim>
class SourceTerm : public Function<dim>
{
public:
    SourceTerm() : Function<dim>(2) {}
    
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
        double x = p[0];
        double y = p[1];
        
        // Source term for subdomain B
        values(0) = (-101*x*x*y*y*y/1470 - 202*x*x*y*y/1225 + 1111*x*x*y/4900 - 101*x*x/2450 
                     + 11869*x*y*y*y/18375 + 47476*x*y*y/30625 - 130559*x*y/61250 + 11869*x/30625 
                     - 101*y*y*y*y*y/6125 - 404*y*y*y*y/6125 - 9591*y*y*y/39200 - 274761*y*y/245000 
                     + 2755731*y/1960000 - 250521/980000);
        values(1) = (33183*x*x*y*y/24500 + 66366*x*x*y/30625 - 365013*x*x/245000 
                     - 101*x*y*y*y*y/2100 - 404*x*y*y*y/2625 - 17141*x*y*y/9800 - 104794*x*y/30625 
                     + 1113849*x/490000 + 499*y*y*y*y/3675 + 7984*y*y*y/18375 - 41347*y*y/49000 
                     + 2509*y/6125 - 5643/98000);
        
        for (unsigned int d = 2; d < dim; ++d)
            values(d) = 0.0;
    }
};

template<int dim>
class ElasticityProblem
{
public:
    ElasticityProblem(double lambda, double mu, unsigned int refinement_level);
    void solve();
    void write_solution(const std::string &filename);
    void write_interface_data(const std::string &filename);
    unsigned int get_ndof() const { return dof_handler.n_dofs(); }
    
private:
    void assemble_system();
    void apply_boundary_conditions();
    
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> system_rhs;
    Vector<double> solution;
    
    const double lambda;
    const double mu;
    
    FESystem<dim> fe;
    Quadrature<dim>   q_point;
    Quadrature<dim-1> q_boundary;
    
    FEValues<dim> fe_values;
    FEFaceValues<dim> fe_face_values;
    
    std::vector<unsigned int> local_dof_indices;
    
    // Interface position
    const double x_interface;
};

template<int dim>
ElasticityProblem<dim>::ElasticityProblem(double lambda_, double mu_, unsigned int refinement_level)
    : lambda(lambda_), mu(mu_), x_interface(5.0/8.0)
{
    // Create mesh for subdomain B: (0.625, 1.5) x (0, 1)
    Point<dim> p1(x_interface, 0, 0);
    Point<dim> p2(1.5, 1, 0);
    GridGenerator::hyper_rectangle(triangulation, p1, p2);
    triangulation.refine_global(refinement_level);
    
    // Finite element: Q1 for each component
    fe = FESystem<dim>(FE_Q<dim>(1), dim);
    
    dof_handler.distribute_dofs(fe);
    
    // Quadrature
    q_point = Quadrature<dim>(3);
    q_boundary = Quadrature<dim-1>(3);
    
    fe_values.initialize(q_point);
    fe_face_values.initialize(q_boundary);
}

template<int dim>
void ElasticityProblem<dim>::assemble_system()
{
    system_matrix.reinit(0, 0);
    system_rhs.reinit(dof_handler.n_dofs());
    
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    
    std::vector<Tensor<2,dim>> strain_values(q_point.size());
    std::vector<double> weights(q_point.size());
    
    Vector<double> cell_rhs(fe.n_dofs_per_cell());
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_quadrature_weights(weights);
        
        cell->get_dof_indices(local_dof_indices);
        
        cell_rhs = 0;
        
        for (unsigned int q = 0; q < q_point.size(); ++q)
        {
            // Body force contribution
            Vector<double> f(2);
            SourceTerm<dim>().vector_value(fe_values.quadrature_point(q), f);
            
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                for (unsigned int d = 0; d < dim; ++d)
                    cell_rhs(i) -= f(d) * fe_values[i](q)[d] * weights[q];
        }
        
        cell->add_contributions(system_matrix, cell_rhs, local_dof_indices);
    }
    
    DoFTools::add_contributions(system_matrix, system_rhs, local_dof_indices);
}

template<int dim>
void ElasticityProblem<dim>::apply_boundary_conditions()
{
    std::vector<bool> constrained_dofs(dof_handler.n_dofs(), false);
    
    // Mark constrained dofs (Dirichlet boundaries)
    for (const auto &cell : dof_handler.active_cell_iterators())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f))
            {
                const Point<dim> center = cell->face(f)->center();
                
                // Bottom (y=0), Top (y=1), Right (x=1.5) are Dirichlet
                // Left (x=0.625) is Neumann (interface)
                if (std::abs(center[1]) < 1e-10 || std::abs(center[1] - 1.0) < 1e-10 || 
                    std::abs(center[0] - 1.5) < 1e-10)
                {
                    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
                    {
                        const unsigned int vertex_index = cell->face(f)->vertex_index(v);
                        const auto vertex_dofs = dof_handler.vertex_dof_indices(vertex_index);
                        for (const auto &dof : vertex_dofs)
                            constrained_dofs[dof] = true;
                    }
                }
            }
    
    DoFTools::zero_entries(system_matrix, constrained_dofs);
    DoFTools::zero_entries(system_rhs, constrained_dofs);
    
    // Set diagonal to 1 for constrained dofs
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
        if (constrained_dofs[i])
        {
            system_matrix.add(i, i, 1.0);
            system_rhs(i) = 0.0;
        }
}

template<int dim>
void ElasticityProblem<dim>::solve()
{
    assemble_system();
    apply_boundary_conditions();
    
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 0.5);
    
    solution = 0;
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    
    std::cout << "Conjugate gradient iterations: " << solver_control.last_step() << std::endl;
}

template<int dim>
void ElasticityProblem<dim>::write_solution(const std::string &filename)
{
    std::ofstream out(filename);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const Point<dim> p = cell->vertex(v);
            const auto vertex_dofs = dof_handler.vertex_dof_indices(v);
            
            double ux = 0, uy = 0;
            for (unsigned int d = 0; d < dim; ++d)
            {
                const unsigned int dof_idx = vertex_dofs[d];
                if (dof_idx < solution.size())
                {
                    if (d == 0) ux = solution(dof_idx);
                    else if (d == 1) uy = solution(dof_idx);
                }
            }
            
            out << p[0] << "," << p[1] << "," << ux << "," << uy << std::endl;
        }
    
    out.close();
}

template<int dim>
void ElasticityProblem<dim>::write_interface_data(const std::string &filename)
{
    std::ofstream out(filename);
    
    // Write interface displacement and traction
    for (int i = 0; i < 44; ++i)
    {
        double y = 0.25 + (i + 0.5) * 0.5 / 44;
        double x = x_interface;
        
        // For simplicity, just output zeros for now
        out << x << "," << y << ",0,0,0,0" << std::endl;
    }
    
    out.close();
}

int main()
{
    try
    {
        const unsigned int refinement_level = 3;
        const double lambda = 500.0;
        const double mu = 1250.0;
        
        ElasticityProblem<2> problem(lambda, mu, refinement_level);
        problem.solve();
        problem.write_solution("solution_B.txt");
        problem.write_interface_data("interface_B.csv");
        
        std::cout << "NDOF = " << problem.get_ndof() << std::endl;
        std::cout << "Solution written to solution_B.txt" << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }
    
    return 0;
}
