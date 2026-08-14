/* deal.II path-walk participant: scalar diffusion OR plane-strain elasticity,
 * either Dirichlet/Neumann role, interface along x or y, 2-D.
 *
 * WHAT DOES NOT SHIP FOR THIS. heat_iface_dealii.cc is scalar conduction with a
 * CONSTANT source and a straight interface at x = const; elast_iface_dealii.cc
 * is its vector twin, also with no volumetric load. A manufactured problem needs
 * a source that varies over the domain, which arrives here as a muparser
 * expression and is evaluated at the quadrature points, and the conjugate-heat
 * cell needs a full conductivity TENSOR, which neither of them carries.
 *
 * FLUX / TRACTION RECOVERY. The Dirichlet side exports the CONSISTENT (reaction)
 * flux: int_Gamma qn phi_i ds = -(A u - b)_i with A and b assembled with NO
 * boundary condition applied, so the constrained rows still carry the reaction.
 * An L2 projection of the gradient is only O(h) accurate ON the boundary -- the
 * superconvergence points are interior -- and the boundary trace is exactly what
 * the coupling reads. Measured elsewhere in this campaign: projection gives a
 * coupled field order that DRIFTS DOWN under refinement (1.76 then 1.59, below
 * the [1.6, 2.4] pass band), the consistent recovery gives ~1.95 and rising.
 *
 * SIGN CONVENTION, identical to every other participant here:
 *     q_out = -(K grad u) . n_own      (scalar)
 *     q_out = -(sigma . n_own)         (vector)
 * so the two sides export opposite signs and the Neumann side applies the
 * partner's numbers unchanged.
 *
 * THE TWO INTERFACE ENDS belong to the OUTER boundary on BOTH sides: they lie on
 * an outer face too and carry the prescribed datum in the un-split problem.
 * Giving them to the interface leaves the two subproblems disagreeing there by
 * O(1) and the partitioned residual never falls. They are still exported.
 *
 * Input file (argv[1]), whitespace separated:
 *   mode(0=scalar,1=vector,2=transient-scalar) side(0=dirichlet,1=neumann)
 *   axis xi
 *   x0 x1 y0 y1 nx ny
 *   k00 k01 k10 k11            (scalar)  |  lambda mu 0 0   (vector)
 *                              (mode 2: k00 = k, k01 = dt, k10 = nsteps)
 *   <source expression, one line>        (scalar: f(x,y[,t]); vector: fx, fy)
 *   n_samples
 *   s_0 v0_0 [v1_0 ...]        (mode 2: nsteps values per interface point)
 *   ...
 * Output (argv[2]): one line "s u... q..." per interface node, ordered by the
 * free coordinate (mode 2: nsteps u-slots then nsteps q-slots); then "#NODES"
 * followed by "x y u[...]" for all nodes (mode 2: the FINAL-TIME field).
 *
 * MODE 2 IS THE TRANSIENT PARTICIPANT THAT DOES NOT SHIP. heat_iface_dealii.cc
 * is steady; D8's walk had a transient FEniCSx side and none for deal.II. The
 * scheme is Crank-Nicolson, and the exchange is WAVEFORM relaxation: each
 * coupling iteration integrates the WHOLE time window and the exchanged object
 * is the entire space-time interface trace -- n points with nsteps components
 * -- which the driver moves and relaxes unchanged because it treats values as
 * opaque numbers on coordinates. The flux convention both sides share: slot s
 * carries the STEP-AVERAGED flux of step s, exported from the step residual as
 * -r_i/(w_i*dt) and applied by the partner as dt*q_s*phi_i. Step-average vs
 * midpoint differs by O(dt^2), so Crank-Nicolson accuracy is preserved.
 */
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
// FEFaceValues lives in fe_values.h from 9.8 on; there is no
// deal.II/fe/fe_face_values.h in this install.
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

using namespace dealii;
static const unsigned int DIM = 2;

struct Cfg {
  int mode, side, axis;
  double xi, x0, x1, y0, y1;
  unsigned nx, ny;
  double m[4];
  std::string src0, src1;
  std::vector<double> s, v0, v1;
  std::vector<std::vector<double>> trace;   // mode 2: per point, nsteps values
};

static Cfg read_cfg(const std::string &path) {
  std::ifstream in(path);
  Cfg c;
  in >> c.mode >> c.side >> c.axis >> c.xi;
  in >> c.x0 >> c.x1 >> c.y0 >> c.y1 >> c.nx >> c.ny;
  for (int i = 0; i < 4; ++i) in >> c.m[i];
  std::string dummy;
  std::getline(in, dummy);
  std::getline(in, c.src0);
  std::getline(in, c.src1);
  int n = 0;
  in >> n;
  c.s.resize(n); c.v0.resize(n); c.v1.assign(n, 0.0);
  const int nsteps = (c.mode == 2) ? int(c.m[2]) : 0;
  if (c.mode == 2) c.trace.assign(n, std::vector<double>(nsteps, 0.0));
  for (int i = 0; i < n; ++i) {
    in >> c.s[i];
    if (c.mode == 2)
      for (int k = 0; k < nsteps; ++k) in >> c.trace[i][k];
    else {
      in >> c.v0[i];
      if (c.mode == 1) in >> c.v1[i];
    }
  }
  return c;
}

// linear interpolation of a whole imported TRACE column onto a coordinate
static double interp_col(const Cfg &c, int col, double q) {
  if (c.s.empty()) return 0.0;
  auto val = [&](std::size_t j) { return c.trace[j][col]; };
  if (q <= c.s.front()) return val(0);
  if (q >= c.s.back()) return val(c.s.size() - 1);
  auto it = std::lower_bound(c.s.begin(), c.s.end(), q);
  const std::size_t j = static_cast<std::size_t>(it - c.s.begin());
  const double t = (q - c.s[j - 1]) / (c.s[j] - c.s[j - 1]);
  return (1.0 - t) * val(j - 1) + t * val(j);
}

// linear interpolation of the partner's samples onto a coordinate
static double interp(const std::vector<double> &s, const std::vector<double> &v,
                     double q) {
  if (s.empty()) return 0.0;
  if (q <= s.front()) return v.front();
  if (q >= s.back()) return v.back();
  auto it = std::lower_bound(s.begin(), s.end(), q);
  const std::size_t j = static_cast<std::size_t>(it - s.begin());
  const double t = (q - s[j - 1]) / (s[j] - s[j - 1]);
  return (1.0 - t) * v[j - 1] + t * v[j];
}

int main(int argc, char **argv) {
  if (argc < 3) { std::cerr << "usage: prog in out\n"; return 2; }
  const Cfg c = read_cfg(argv[1]);
  const unsigned ncomp = (c.mode == 1) ? DIM : 1;
  const double tol = 1e-9;

  Triangulation<DIM> tria;
  GridGenerator::subdivided_hyper_rectangle(
      tria, {c.nx, c.ny}, Point<DIM>(c.x0, c.y0), Point<DIM>(c.x1, c.y1));

  FESystem<DIM> fe(FE_Q<DIM>(1), ncomp);
  DoFHandler<DIM> dh(tria);
  dh.distribute_dofs(fe);
  const unsigned ndofs = dh.n_dofs();

  std::vector<Point<DIM>> support(ndofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<DIM>(), dh, support);

  // WHICH COMPONENT EACH GLOBAL DOF CARRIES. Taking it as `i % ncomp` assumes a
  // global numbering deal.II does not promise; read it off the cell-local map
  // instead, which is exact for any numbering and any element.
  std::vector<unsigned> dof_comp(ndofs, 0);
  {
    std::vector<types::global_dof_index> tmp(fe.n_dofs_per_cell());
    for (const auto &cell : dh.active_cell_iterators()) {
      cell->get_dof_indices(tmp);
      for (unsigned i = 0; i < tmp.size(); ++i)
        dof_comp[tmp[i]] = fe.system_to_component_index(i).first;
    }
  }

  auto on_iface = [&](const Point<DIM> &p) {
    return std::fabs(p[c.axis] - c.xi) < tol;
  };
  auto on_outer = [&](const Point<DIM> &p) {
    const double lo[2] = {c.x0, c.y0}, hi[2] = {c.x1, c.y1};
    for (unsigned a = 0; a < DIM; ++a)
      for (double val : {lo[a], hi[a]}) {
        if (a == (unsigned)c.axis && std::fabs(val - c.xi) < tol) continue;
        if (std::fabs(p[a] - val) < tol) return true;
      }
    return false;
  };

  // ══════════════════════════════════════════════════════════════════
  // MODE 2: transient scalar conduction, Crank-Nicolson, waveform exchange
  // ══════════════════════════════════════════════════════════════════
  if (c.mode == 2) {
    const double kappa = c.m[0], dt = c.m[1];
    const int nsteps = int(c.m[2]);
    FunctionParser<DIM> rhs_t(1);
    rhs_t.initialize("x,y,t", c.src0,
                     std::map<std::string, double>(), /*time_dependent=*/true);

    DynamicSparsityPattern dsp(ndofs);
    DoFTools::make_sparsity_pattern(dh, dsp);
    SparsityPattern sp;
    sp.copy_from(dsp);
    SparseMatrix<double> M(sp), K(sp), A(sp), Aun(sp);
    const QGauss<DIM> quad(4);
    const QGauss<DIM - 1> fquad(4);
    FEValues<DIM> fev(fe, quad, update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);
    FEFaceValues<DIM> ffv(fe, fquad, update_values |
                                         update_quadrature_points |
                                         update_JxW_values);
    const unsigned dpc = fe.n_dofs_per_cell();
    FullMatrix<double> me(dpc, dpc), ke(dpc, dpc);
    std::vector<types::global_dof_index> ldi(dpc);
    for (const auto &cell : dh.active_cell_iterators()) {
      fev.reinit(cell);
      me = 0;
      ke = 0;
      for (unsigned q = 0; q < quad.size(); ++q)
        for (unsigned i = 0; i < dpc; ++i)
          for (unsigned j = 0; j < dpc; ++j) {
            me(i, j) += fev.shape_value(i, q) * fev.shape_value(j, q) *
                        fev.JxW(q);
            ke(i, j) += kappa * fev.shape_grad(i, q) * fev.shape_grad(j, q) *
                        fev.JxW(q);
          }
      cell->get_dof_indices(ldi);
      for (unsigned i = 0; i < dpc; ++i)
        for (unsigned j = 0; j < dpc; ++j) {
          M.add(ldi[i], ldi[j], me(i, j));
          K.add(ldi[i], ldi[j], ke(i, j));
        }
    }
    Aun.copy_from(M);
    Aun.add(dt / 2.0, K);

    // w_i = int_Gamma phi_i ds, assembled exactly on the interface facets
    Vector<double> w(ndofs);
    for (const auto &cell : dh.active_cell_iterators())
      for (const auto &face : cell->face_iterators()) {
        if (!face->at_boundary()) continue;
        if (std::fabs(face->center()[c.axis] - c.xi) > tol) continue;
        ffv.reinit(cell, face);
        cell->get_dof_indices(ldi);
        for (unsigned q = 0; q < fquad.size(); ++q)
          for (unsigned i = 0; i < dpc; ++i)
            w(ldi[i]) += ffv.shape_value(i, q) * ffv.JxW(q);
      }

    auto assemble_F = [&](double tval, Vector<double> &F) {
      F = 0;
      rhs_t.set_time(tval);
      Vector<double> cb(dpc);
      for (const auto &cell : dh.active_cell_iterators()) {
        fev.reinit(cell);
        cb = 0;
        for (unsigned q = 0; q < quad.size(); ++q) {
          const double fq = rhs_t.value(fev.quadrature_point(q));
          for (unsigned i = 0; i < dpc; ++i)
            cb(i) += fev.shape_value(i, q) * fq * fev.JxW(q);
        }
        cell->get_dof_indices(ldi);
        for (unsigned i = 0; i < dpc; ++i) F(ldi[i]) += cb(i);
      }
    };

    std::map<double, unsigned> byfree;   // interface points, one scalar dof
    for (unsigned i = 0; i < ndofs; ++i)
      if (on_iface(support[i]))
        byfree[std::round(support[i][1 - c.axis] * 1e9) / 1e9] = i;

    Vector<double> u(ndofs), unew(ndofs), b(ndofs), bun(ndofs), F0(ndofs),
        F1(ndofs), r(ndofs), tmp(ndofs);
    u = 0;                                       // the initial datum is u = 0
    const std::size_t nif = byfree.size();
    std::vector<std::vector<double>> tr_u(nif, std::vector<double>(nsteps));
    std::vector<std::vector<double>> tr_q(nif, std::vector<double>(nsteps));

    assemble_F(0.0, F1);
    for (int step = 0; step < nsteps; ++step) {
      const double t1 = (step + 1) * dt;
      F0 = F1;
      assemble_F(t1, F1);
      // bun = M u - dt/2 K u + dt/2 (F0 + F1)   (no interface flux term:
      // the residual below must isolate exactly that term)
      M.vmult(bun, u);
      K.vmult(tmp, u);
      bun.add(-dt / 2.0, tmp);
      bun.add(dt / 2.0, F0);
      bun.add(dt / 2.0, F1);
      b = bun;
      if (c.side == 1) {
        // NEUMANN: the partner's step-s flux density, integrated exactly on
        // the interface facets and applied over the whole step: dt * q_s.
        for (const auto &cell : dh.active_cell_iterators())
          for (const auto &face : cell->face_iterators()) {
            if (!face->at_boundary()) continue;
            if (std::fabs(face->center()[c.axis] - c.xi) > tol) continue;
            ffv.reinit(cell, face);
            cell->get_dof_indices(ldi);
            for (unsigned q = 0; q < fquad.size(); ++q) {
              const double sfree = ffv.quadrature_point(q)[1 - c.axis];
              const double qs = interp_col(c, step, sfree);
              for (unsigned i = 0; i < dpc; ++i)
                b(ldi[i]) += dt * ffv.shape_value(i, q) * qs * ffv.JxW(q);
            }
          }
      }
      std::map<types::global_dof_index, double> bvals;
      for (unsigned i = 0; i < ndofs; ++i) {
        if (on_outer(support[i]))
          bvals[i] = 0.0;                        // u = 0 outside at all times
        else if (c.side == 0 && on_iface(support[i]))
          bvals[i] = interp_col(c, step, support[i][1 - c.axis]);
      }
      A.copy_from(Aun);
      MatrixTools::apply_boundary_values(bvals, A, unew, b);
      SparseDirectUMFPACK slv;
      slv.initialize(A);
      slv.vmult(unew, b);
      for (const auto &kv : bvals) unew(kv.first) = kv.second;

      if (c.side == 0) {
        // step-averaged consistent flux: r = Aun u^{n+1} - bun on the
        // constrained rows equals -dt * int_Gamma qbar phi_i ds
        Aun.vmult(r, unew);
        r -= bun;
        std::size_t k = 0;
        for (const auto &kv : byfree) {
          const unsigned i = kv.second;
          tr_q[k][step] = (std::fabs(w(i)) > 1e-14)
                              ? -r(i) / (w(i) * dt)
                              : 0.0;
          ++k;
        }
      }
      std::size_t k = 0;
      for (const auto &kv : byfree) {
        tr_u[k][step] = unew(kv.second);
        ++k;
      }
      u = unew;
    }

    // the two interface ENDS lie on the outer boundary too; their rows carry
    // the outer reaction as well, so copy the nearest interior value
    {
      std::vector<unsigned> idx;
      std::vector<bool> bad;
      for (const auto &kv : byfree) {
        idx.push_back(kv.second);
        bad.push_back(on_outer(support[kv.second]));
      }
      for (std::size_t k = 0; k < idx.size(); ++k)
        if (bad[k]) {
          std::size_t g = k;
          for (std::size_t d = 1; d < idx.size(); ++d) {
            if (k >= d && !bad[k - d]) { g = k - d; break; }
            if (k + d < idx.size() && !bad[k + d]) { g = k + d; break; }
          }
          tr_q[k] = tr_q[g];
        }
    }

    std::ofstream out(argv[2]);
    out.precision(17);
    std::size_t k = 0;
    for (const auto &kv : byfree) {
      out << kv.first;
      for (int st = 0; st < nsteps; ++st) out << " " << tr_u[k][st];
      for (int st = 0; st < nsteps; ++st) out << " " << tr_q[k][st];
      out << "\n";
      ++k;
    }
    out << "#NODES\n";
    for (unsigned i = 0; i < ndofs; ++i)
      out << support[i][0] << " " << support[i][1] << " " << u(i) << "\n";
    out << "#NDOF " << ndofs << "\n";
    std::cout << "Number of active cells: " << tria.n_active_cells() << "\n"
              << "Number of degrees of freedom: " << ndofs << "\n"
              << "Steps: " << nsteps << " dt " << dt << "\n";
    return 0;
  }

  // ── the source, as a muparser expression evaluated at the quadrature points
  std::vector<std::string> exprs;
  exprs.push_back(c.src0);
  if (c.mode == 1) exprs.push_back(c.src1);
  FunctionParser<DIM> rhs(ncomp);
  {
    std::string joined = exprs[0];
    for (unsigned i = 1; i < exprs.size(); ++i) joined += ";" + exprs[i];
    rhs.initialize("x,y", joined, {});
  }

  AffineConstraints<double> constraints;   // outer + (maybe) interface
  AffineConstraints<double> none;          // no constraints at all
  none.close();

  for (unsigned i = 0; i < ndofs; ++i) {
    const bool outer = on_outer(support[i]);
    const bool iface = on_iface(support[i]);
    if (outer) {
      constraints.add_line(i);
      constraints.set_inhomogeneity(i, 0.0);
    } else if (iface && c.side == 0) {
      const double sfree = support[i][1 - c.axis];
      const unsigned cc = dof_comp[i];
      const double val = (cc == 0) ? interp(c.s, c.v0, sfree)
                                   : interp(c.s, c.v1, sfree);
      constraints.add_line(i);
      constraints.set_inhomogeneity(i, val);
    }
  }
  constraints.close();

  // THE PATTERN MUST KEEP THE CONSTRAINED ROWS. Building it with constraints
  // and keep_constrained_dofs=false leaves those rows empty, and the
  // unconstrained assembly below then writes into entries that do not exist:
  // silent in 2-D, a heap corruption and SIGSEGV inside UMFPACK in 3-D. That
  // exact failure is on record in path_readiness.json from the D2 walk.
  DynamicSparsityPattern dsp(ndofs);
  DoFTools::make_sparsity_pattern(dh, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp), Aun(sp);
  Vector<double> b(ndofs), bun(ndofs), u(ndofs);

  const QGauss<DIM> quad(4);
  const QGauss<DIM - 1> fquad(4);
  FEValues<DIM> fev(fe, quad,
                    update_values | update_gradients | update_quadrature_points |
                        update_JxW_values);
  FEFaceValues<DIM> ffv(fe, fquad,
                        update_values | update_quadrature_points |
                            update_JxW_values);
  const unsigned dpc = fe.n_dofs_per_cell();
  FullMatrix<double> ce(dpc, dpc);
  Vector<double> cb(dpc);
  std::vector<types::global_dof_index> ldi(dpc);

  for (const auto &cell : dh.active_cell_iterators()) {
    fev.reinit(cell);
    ce = 0; cb = 0;
    for (unsigned q = 0; q < quad.size(); ++q) {
      const Point<DIM> &p = fev.quadrature_point(q);
      Vector<double> fv(ncomp);
      rhs.vector_value(p, fv);
      for (unsigned i = 0; i < dpc; ++i) {
        const unsigned ci = fe.system_to_component_index(i).first;
        for (unsigned j = 0; j < dpc; ++j) {
          const unsigned cj = fe.system_to_component_index(j).first;
          double v = 0.0;
          if (c.mode == 0) {
            const Tensor<1, DIM> gi = fev.shape_grad(i, q);
            const Tensor<1, DIM> gj = fev.shape_grad(j, q);
            for (unsigned a = 0; a < DIM; ++a)
              for (unsigned bb = 0; bb < DIM; ++bb)
                v += c.m[a * DIM + bb] * gj[bb] * gi[a];
          } else {
            const double lam = c.m[0], mu = c.m[1];
            const Tensor<1, DIM> gi = fev.shape_grad(i, q);
            const Tensor<1, DIM> gj = fev.shape_grad(j, q);
            // sigma(u_j) : eps(v_i), with u_j = e_cj phi_j, v_i = e_ci phi_i
            v += lam * gj[cj] * gi[ci];
            v += mu * gj[ci] * gi[cj];
            if (ci == cj)
              for (unsigned a = 0; a < DIM; ++a) v += mu * gj[a] * gi[a];
          }
          ce(i, j) += v * fev.JxW(q);
        }
        cb(i) += fev.shape_value(i, q) * fv[ci] * fev.JxW(q);
      }
    }
    // NEUMANN: the partner's number, applied UNCHANGED on the interface facets
    if (c.side == 1) {
      for (const auto &face : cell->face_iterators()) {
        if (!face->at_boundary()) continue;
        if (std::fabs(face->center()[c.axis] - c.xi) > tol) continue;
        ffv.reinit(cell, face);
        for (unsigned q = 0; q < fquad.size(); ++q) {
          const double sfree = ffv.quadrature_point(q)[1 - c.axis];
          const double q0 = interp(c.s, c.v0, sfree);
          const double q1 = interp(c.s, c.v1, sfree);
          for (unsigned i = 0; i < dpc; ++i) {
            const unsigned ci = fe.system_to_component_index(i).first;
            cb(i) += ffv.shape_value(i, q) * (ci == 0 ? q0 : q1) * ffv.JxW(q);
          }
        }
      }
    }
    cell->get_dof_indices(ldi);
    // the UNCONSTRAINED system, kept for the reaction
    none.distribute_local_to_global(ce, cb, ldi, Aun, bun);
    constraints.distribute_local_to_global(ce, cb, ldi, A, b);
  }

  SparseDirectUMFPACK solver;
  solver.initialize(A);
  u = b;
  solver.vmult(u, b);
  constraints.distribute(u);

  // ── the interface nodes, ordered by the free coordinate ─────────────
  std::map<double, std::vector<unsigned>> byfree;
  for (unsigned i = 0; i < ndofs; ++i)
    if (on_iface(support[i]))
      byfree[std::round(support[i][1 - c.axis] * 1e9) / 1e9].push_back(i);

  // w_i = int_Gamma phi_i ds, assembled exactly on the interface facets
  Vector<double> w(ndofs);
  for (const auto &cell : dh.active_cell_iterators())
    for (const auto &face : cell->face_iterators()) {
      if (!face->at_boundary()) continue;
      if (std::fabs(face->center()[c.axis] - c.xi) > tol) continue;
      ffv.reinit(cell, face);
      cell->get_dof_indices(ldi);
      for (unsigned q = 0; q < fquad.size(); ++q)
        for (unsigned i = 0; i < dpc; ++i)
          w(ldi[i]) += ffv.shape_value(i, q) * ffv.JxW(q);
    }

  Vector<double> res(ndofs);
  Aun.vmult(res, u);
  res -= bun;

  std::ofstream out(argv[2]);
  out.precision(17);
  for (const auto &kv : byfree) {
    std::vector<double> uu(ncomp, 0.0), qq(ncomp, 0.0);
    for (unsigned i : kv.second) {
      const unsigned cc = dof_comp[i];
      uu[cc] = u(i);
      if (c.side == 0)
        qq[cc] = (std::fabs(w(i)) > 1e-14) ? -res(i) / w(i) : 0.0;
    }
    out << kv.first;
    for (unsigned k = 0; k < ncomp; ++k) out << " " << uu[k];
    for (unsigned k = 0; k < ncomp; ++k) out << " " << qq[k];
    out << "\n";
  }
  out << "#NODES\n";
  std::map<std::pair<double, double>, std::vector<double>> nodemap;
  for (unsigned i = 0; i < ndofs; ++i) {
    const auto key = std::make_pair(std::round(support[i][0] * 1e9) / 1e9,
                                    std::round(support[i][1] * 1e9) / 1e9);
    auto &vv = nodemap[key];
    if (vv.empty()) vv.assign(ncomp, 0.0);
    vv[dof_comp[i]] = u(i);
  }
  for (const auto &kv : nodemap) {
    out << kv.first.first << " " << kv.first.second;
    for (double vv : kv.second) out << " " << vv;
    out << "\n";
  }
  out << "#NDOF " << ndofs << "\n";
  std::cout << "Number of active cells: " << tria.n_active_cells() << "\n"
            << "Number of degrees of freedom: " << ndofs << "\n";
  return 0;
}
