// ---------------------------------------------------------------
// Simplified deal.II participant for coupled anisotropic heat conduction
// Debug version with more error checking
// ---------------------------------------------------------------
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
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
#include <iomanip>
#include <map>

using namespace dealii;

class SourceA : public Function<2>
{
public:
    SourceA() : Function<2>(2) {}
    
    virtual double value(const Point<2>& p, const unsigned int = 0) const override
    {
        double x = p(0);
        double y = p(1);
        return (9.0*x*x*x*y + 2.0*x*x*x + 27.0*x*x*y*y/4.0 + 129.0*x*x*y/16.0 - 319.0*x*x/24.0
                + 9.0*x*y*y*y/2.0 + 177.0*x*y*y/32.0 - 631.0*x*y/24.0 + 323.0*x/32.0
                + 27.0*y*y*y/32.0 - 85.0*y*y/12.0 + 673.0*y/96.0 - 37.0/48.0);
    }
};

int main()
{
    try
    {
        std::cerr << "Starting deal.II solver..." << std::endl;
        
        // Read input file
        std::ifstream infile("dealii_input.txt");
        if (!infile.is_open())
        {
            std::cerr << "Cannot open dealii_input.txt" << std::endl;
            return 1;
        }
        
        unsigned int side_flag;
        double X0, X1, Y0, Y1, IFACE_X;
        unsigned int NX, NY, DEGREE;
        
        infile >> side_flag >> X0 >> X1 >> Y0 >> Y1 >> IFACE_X >> NX >> NY >> DEGREE;
        std::cerr << "Read params: side=" << side_flag << " X=[" << X0 << "," << X1 
                  << "] Y=[" << Y0 << "," << Y1 << "] iface=" << IFACE_X
                  << " NX=" << NX << " NY=" << NY << " deg=" << DEGREE << std::endl;
        
        unsigned int n_samples;
        infile >> n_samples;
        std::vector<std::pair<double, double>> interface_samples(n_samples);
        for (unsigned int i = 0; i < n_samples; ++i)
        {
            infile >> interface_samples[i].first >> interface_samples[i].second;
        }
        infile.close();
        
        std::cerr << "Read " << n_samples << " interface samples" << std::endl;
        
        const unsigned int dim = 2;
        const double K00 = 1.0, K01 = 0.5, K10 = 0.5, K11 = 2.0;
        
        // Build mesh using generate_subdivided_hyper_rectangle
        Triangulation<dim> tria;
        std::vector<unsigned int> repetitions = {NX, NY};
        tria.generate_subdivided_hyper_rectangle(repetitions, Point<dim>(X0, Y0), Point<dim>(X1, Y1), true);
        
        std::cerr << "Mesh created: " << tria.n_active_cells() << " cells" << std::endl;
        
        FE_Q<dim> fe(DEGREE);
        DoFHandler<dim> dof_handler(tria);
        dof_handler.distribute_dofs(fe);
        
        std::cerr << "DOFs distributed: " << dof_handler.n_dofs() << " DOFs" << std::endl;
        
        const types::boundary_id interface_id = 1;  // right boundary
        
        AffineConstraints<double> constraints;
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        
        Functions::ZeroFunction<dim> zero_func;
        
        if (side_flag == 0)
        {
            for (types::boundary_id bid : tria.get_boundary_ids())
                VectorTools::interpolate_boundary_values(dof_handler, bid, zero_func, constraints);
        }
        else
        {
            VectorTools::interpolate_boundary_values(dof_handler, 0, zero_func, constraints);
            VectorTools::interpolate_boundary_values(dof_handler, 2, zero_func, constraints);
            VectorTools::interpolate_boundary_values(dof_handler, 3, zero_func, constraints);
        }
        constraints.close();
        
        std::cerr << "Constraints closed: " << constraints.n_constraints() << " constrained DOFs" << std::endl;
        
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
        SparsityPattern sparsity;
        sparsity.copy_from(dsp);
        SparseMatrix<double> A(sparsity);
        Vector<double> u(dof_handler.n_dofs()), b(dof_handler.n_dofs());
        
        QGauss<dim> quadrature(DEGREE + 1);
        FEValues<dim> fe_values(fe, quadrature,
                               update_values | update_gradients | 
                               update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        FullMatrix<double> cell_A(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_b(dofs_per_cell);
        std::vector<types::global_dof_index> local_dofs(dofs_per_cell);
        
        SourceA source_func;
        
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            cell_A = 0.;
            cell_b = 0.;
            fe_values.reinit(cell);
            
            for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
                double JxW = fe_values.JxW(q);
                double f_val = source_func.value(fe_values.quadrature_point(q), 0);
                
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    cell_b(i) += f_val * fe_values.shape_value(i, q) * JxW;
                    
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        double contrib = (K00 * fe_values.shape_grad(i,q)[0] * fe_values.shape_grad(j,q)[0]
                                        + K01 * fe_values.shape_grad(i,q)[0] * fe_values.shape_grad(j,q)[1]
                                        + K10 * fe_values.shape_grad(i,q)[1] * fe_values.shape_grad(j,q)[0]
                                        + K11 * fe_values.shape_grad(i,q)[1] * fe_values.shape_grad(j,q)[1]);
                        cell_A(i, j) += contrib * JxW;
                    }
                }
            }
            
            cell->get_dof_indices(local_dofs);
            constraints.distribute_local_to_global(cell_A, cell_b, local_dofs, A, b);
        }
        
        // Neumann contribution
        if (side_flag == 1 && n_samples > 0)
        {
            QGauss<dim-1> face_quadrature(DEGREE + 1);
            FEFaceValues<dim> fe_face_values(fe, face_quadrature,
                                            update_quadrature_points | update_JxW_values);
            
            std::vector<types::global_dof_index> local_face_dofs(fe.n_dofs_per_face());
            
            for (const auto &cell : dof_handler.active_cell_iterators())
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == interface_id)
                    {
                        fe_face_values.reinit(cell, f);
                        cell->face(f)->get_dof_indices(local_face_dofs);
                        
                        for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                        {
                            double y = fe_face_values.quadrature_point(q)(1);
                            
                            double flux = 0;
                            for (size_t i = 0; i < n_samples - 1; ++i)
                            {
                                if (y >= interface_samples[i].first && y <= interface_samples[i+1].first)
                                {
                                    double t = (y - interface_samples[i].first) / 
                                               (interface_samples[i+1].first - interface_samples[i].first);
                                    flux = (1-t) * interface_samples[i].second + t * interface_samples[i+1].second;
                                    break;
                                }
                            }
                            
                            for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
                            {
                                b(local_face_dofs[i]) += flux * fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);
                            }
                        }
                    }
        }
        
        A.compress(VectorOperation::add);
        b.compress(VectorOperation::add);
        
        std::cerr << "Assembling complete, solving..." << std::endl;
        
        SolverControl control(2000, 1e-12);
        SolverCG<Vector<double>> solver(control);
        PreconditionSSOR<SparseMatrix<double>> prec;
        prec.initialize(A);
        solver.solve(A, u, b, prec);
        
        std::cerr << "Solver converged in " << control.last_step() << " iterations" << std::endl;
        
        constraints.distribute(u);
        
        // Write NDOF
        {
            std::ofstream logfile("ndof.txt");
            logfile << "NDOF = " << dof_handler.n_dofs() << std::endl;
            logfile.close();
        }
        
        // Extract interface data
        std::map<double, std::pair<double, double>> interface_map;
        const double corner_tol = 1e-9;
        
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
            cell->get_dof_indices(cell_dofs);
            
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
            {
                Point<dim> p = cell->vertex(v);
                double x = p(0);
                double y = p(1);
                
                bool on_interface = (std::abs(x - IFACE_X) < 1e-10);
                bool is_corner = (std::abs(y - Y0) < corner_tol || std::abs(y - Y1) < corner_tol);
                
                if (on_interface && !is_corner)
                {
                    types::global_dof_index dof_idx = cell_dofs[v];
                    double T_val = u(dof_idx);
                    
                    double qn = 0;
                    if (side_flag == 0)
                    {
                        double r_i = 0;
                        for (SparseMatrix<double>::iterator it = A.begin(dof_idx); it != 0; ++it)
                            r_i += it->value() * u(it->column());
                        r_i -= b(dof_idx);
                        
                        double h_y = (Y1 - Y0) / NY;
                        if (std::abs(h_y) > 1e-14)
                            qn = -r_i / h_y;
                    }
                    
                    double y_key = std::round(y * 1e12) / 1e12;
                    if (interface_map.find(y_key) == interface_map.end())
                    {
                        interface_map[y_key] = {T_val, qn};
                    }
                    else
                    {
                        interface_map[y_key].first = (interface_map[y_key].first + T_val) / 2.0;
                        interface_map[y_key].second = (interface_map[y_key].second + qn) / 2.0;
                    }
                }
            }
        }
        
        std::vector<std::tuple<double, double, double>> interface_data;
        for (const auto& entry : interface_map)
        {
            interface_data.push_back(std::make_tuple(entry.first, entry.second.first, entry.second.second));
        }
        std::sort(interface_data.begin(), interface_data.end());
        
        std::ofstream outfile("dealii_output.txt");
        outfile << std::fixed << std::setprecision(16);
        for (const auto &data : interface_data)
        {
            outfile << std::get<0>(data) << " " 
                    << std::get<1>(data) << " " 
                    << std::get<2>(data) << "\n";
        }
        outfile.close();
        
        std::cout << "NDOF = " << dof_handler.n_dofs() << std::endl;
        std::cout << "Active cells = " << tria.n_active_cells() << std::endl;
        std::cout << "Interface points = " << interface_data.size() << std::endl;
        
        return 0;
    }
    catch (std::exception &exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }
}
