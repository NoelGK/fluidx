import numpy as np
import tqdm

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (
    Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, 
    form, locate_dofs_topological, set_bc, locate_dofs_geometrical
)

from dolfinx.fem.petsc import (
    apply_lifting, assemble_matrix, assemble_vector, 
    create_vector, create_matrix, set_bc
)

from dolfinx.io import VTXWriter, gmshio
from dolfinx import plot
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells

from ufl import (
    FiniteElement, VectorElement, TrialFunction, TestFunction, dot, inner, FacetNormal, 
    grad, nabla_grad, div, ds, Measure, dx, lhs, rhs, sym, sqrt, SpatialCoordinate
)

from functions import Sij, nu_T


# =============================================================================
#                 PARÁMETROS Y DOMINIO DE LA SIMULACIÓN
# =============================================================================

gdim, fdim = 2, 1
L, l0, l, H, h = 2.0, 1.0, 0.5, 0.8, 0.4
res = h / 15.0

fluid_tag, wall_tag, inlet_tag, outlet_tag, free_tag = 1, 2, 3, 4, 5
mesh, _, ft = gmshio.read_from_msh("./domains/stepv3.msh", MPI.COMM_WORLD, rank=0, gdim=gdim)
ft.name = 'Facets'

# Parámetros temporales y físicos
t = 0.0
T = 20.0
num_steps = 25000
dt = T / num_steps
dt = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(5.0e-5))  # Viscosidad dinámica
rho = Constant(mesh, PETSc.ScalarType(1.0))   # Densidad
nu = Constant(mesh, PETSc.ScalarType(mu/rho))  # Viscosidad cinemática
f = Constant(mesh, PETSc.ScalarType((0, -9.8)))  # Body forces
Cs = PETSc.ScalarType(0.035)
D = Constant(mesh, PETSc.ScalarType(1.4e-5))
alpha_atm = 0.3
vel0 = 0.5


# =============================================================================
#        ESPACIOS DE FUNCIONES Y FUNCIONES: ELEMENTOS TAYLOR-HOOD
# =============================================================================

vector_element = VectorElement("CG", mesh.ufl_cell(), 2)
scalar_element = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, vector_element)
Q = FunctionSpace(mesh, scalar_element)


# =============================================================================
#                          CONDICIONES DE CONTORNO
# =============================================================================

class InletVelocity():
    def __init__(self, t, alpha=alpha_atm, u0=vel0, z0=H):
        self.t = t
        self.alpha = alpha
        self.u0 = u0
        self.z0 = z0
        self.tol = 1.0e-5
        # self.norm = 1 / (mid - y0) / (y1 - mid)
        

    # def __call__(self, x):
    #     values = np.zeros((gdim, x.shape[1]),dtype=PETSc.ScalarType)
    #     values[0] = 0.1 * self.norm * (x[1] - h) * (H - x[1])
    #     return values
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]),dtype=PETSc.ScalarType)
        values[0] = self.u0 * ((x[1] + self.tol)/self.z0)**self.alpha
        return values


# # # Condiciones de contorno de la velocidad
# Inlet
u_inlet = Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_tag)))

# Walls
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_tag)), V)

# Free
u_free = np.array((vel0, 0), dtype=PETSc.ScalarType)
bcu_free = dirichletbc(u_free, locate_dofs_topological(V, fdim, ft.find(free_tag)), V)
bcu = [bcu_inflow, bcu_walls, bcu_free]

# # # Condiciones de contorno de la presión
# Outlet
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0), locate_dofs_topological(Q, fdim, ft.find(outlet_tag)), Q)
bcp = [bcp_outlet]


# =============================================================================
#                        FORMULACIÓN VARIACIONAL
# =============================================================================

# Funciones para la velocidad
u_tent = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)  # u_tent disponible como función
u_.name = "u"
u_n = Function(V)  # Función para almacenar el timestep anterior u_{n-1}
u_n1 = Function(V)  # u_{n-2}

# Funciones para la presión
p = TrialFunction(Q)
q = TestFunction(Q)
p_ = Function(Q)  # p disponible como función


# # # PASO 1: VELOCIDAD APROXIMADA (NO INCOMPRESIBLE)
F1 = 1/dt * dot(u_tent - u_n, v) * dx + \
    0.5 * inner(dot(1.5*u_n - 0.5*u_n1, nabla_grad(u_tent + u_n)), v) * dx + \
    0.5 * (nu + nu_T(u_n, Cs=Cs, res=res)) * inner(grad(u_tent + u_n), grad(v)) * dx

F1 += - dot(p_, div(v)) * dx - rho * dot(f, v) * dx

a1 = form(lhs(F1))
L1 = form(rhs(F1))
A1 = create_matrix(a1)
b1 = create_vector(L1)

# # # PASO 2: CÁLCULO DE LA PRESIÓN
a2 = form(dot(grad(p), grad(q)) * dx)
L2 = form(-rho/dt * dot(div(u_), q) * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

# PASO 3: INCOMPRESIBILIDAD DE LA VELOCIDAD
a3 = form(dot(u_tent, v) * dx)
L3 = form(dot(u_, v) * dx - dt/rho * dot(grad(p_), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)


# =============================================================================
#        CONTAMINANTE: FUENTE GAUSIANA CENTRADA EN EL CENTRO
#                    DEL CAÑÓN CON VALOR UNIDAD
# =============================================================================

C_space = FunctionSpace(mesh, ("CG", 2))

# Condiciones de contorno
air_dofs = np.concatenate(
    [
        locate_dofs_topological(C_space, fdim, ft.find(tag)) for \
            tag in [free_tag, inlet_tag, outlet_tag]
    ], axis=0
)
bcc = [dirichletbc(PETSc.ScalarType(0.0), air_dofs, C_space)]

c, c_test = TrialFunction(C_space), TestFunction(C_space)
c_n = Function(C_space)
c_n.name = 'c'
f = Function(C_space)


class Source():
    def __init__(self, t):
        self.t = t
    
    def __call__(self, x, w0=5):
        values = np.exp(-((x[0]-L/2)**2 + (x[1]-0.05)**2)/0.05**2) * self._sigmoid()
        return values
    
    def _sigmoid(self, f0=5.0, t0=3.0):
        return 1 - 1/(1 + np.exp(-f0 * (self.t - t0)))


source_function = Source(t)
f.interpolate(source_function)

F4 = 1/dt * dot(c - c_n, c_test) * dx + \
    0.5 * dot(dot(u_n, grad(c + c_n)), c_test) * dx + \
    0.5 * (3.0*nu_T(u_n, Cs=Cs, res=res) + D) * dot(grad(c + c_n), grad(c_test)) * dx
F4 += - dot(f, c_test) * dx
a4 = form(lhs(F4))
L4 = form(rhs(F4))
A4, b4 = create_matrix(a4), create_vector(L4)


# =============================================================================
#                                   SOLVERS
# =============================================================================

# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

# Solver for step 4
solver4 = PETSc.KSP().create(mesh.comm)
solver4.setOperators(A4)
solver4.setType(PETSc.KSP.Type.MINRES)
pc4 = solver4.getPC()
pc4.setType(PETSc.PC.Type.HYPRE)
pc4.setHYPREType("boomeramg")


# =============================================================================
#               Calcular flujo que sale al exterior
# =============================================================================

if mesh.comm.rank == 0:
    c_flux = np.zeros(num_steps//50, dtype=PETSc.ScalarType)

n = FacetNormal(mesh)
d_surf = Measure("dS", domain=mesh, subdomain_data=ft, subdomain_id=6)
flux = form(c_n * inner(u_n, n('+')) * d_surf - D*inner(grad(c_n)('+'), n('+')) * d_surf)


# =============================================================================
#                Obtener concentración en dos puntos
# =============================================================================

dist_to_wall = 0.1
positions_to_evaluate = np.array(
    [[l+dist_to_wall, h/4, 0],
     [l+dist_to_wall, 3*h/4, 0]]
)

tree = BoundingBoxTree(mesh, mesh.geometry.dim)
cell_candidates = compute_collisions(tree, positions_to_evaluate)
colliding_cells = compute_colliding_cells(mesh, cell_candidates, positions_to_evaluate)
lower_cell, upper_cell = colliding_cells.links(0), colliding_cells.links(1)

if mesh.comm.rank == 0:
    c_lower = np.zeros(num_steps//50, dtype=PETSc.ScalarType)
    c_upper = np.zeros(num_steps//50, dtype=PETSc.ScalarType)


# =============================================================================
#                                  LOOP
# =============================================================================

# # # Archivos de salida
# vtx_u = VTXWriter(mesh.comm, "./outputs/step-visco-u.bp", [u_])
# vtx_u.write(t)

# vtx_c = VTXWriter(mesh.comm, "./outputs/step-visco-c.bp", [c_n])
# vtx_c.write(t)

# vtx_p = VTXWriter(mesh.comm, "./outputs/final/step-p.bp", [p_])
# vtx_p.write(t)

progress = tqdm.tqdm(desc="Solving PDE", total=num_steps)
for i in range(num_steps):
    progress.update(1)
    t += dt.value
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)
    
    source_function.t = t
    f.interpolate(source_function)
    
    # PASO 1
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()
    
    # PASO 2
    with b2.localForm() as loc:
        loc.set(0)
    
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()
    
    # PASO 3
    with b3.localForm() as loc:
        loc.set(0)
    
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()
    
    with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
        loc_n.copy(loc_n1)
        loc_.copy(loc_n)
    
    # CONCENTRACIÓN
    A4.zeroEntries()
    assemble_matrix(A4, a4, bcs=bcc)
    A4.assemble()
    with b4.localForm() as loc:
        loc.set(0)
    
    assemble_vector(b4, L4)
    apply_lifting(b4, [a4], [bcc])
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b4, bcc)
    solver4.solve(b4, c_n.vector)
    c_n.x.scatter_forward()
   
    # # # ESCRIBIR EN ARCHIVO
    # if i%50 == 0:
    #     vtx_u.write(t)
    #     vtx_c.write(t)
    #     vtx_p.write(t)
        
    # # # Calcular flujo a través de la superficie
    if i%50 == 0:
        # Flujo al exterior del cañón
        flux_processor0 = mesh.comm.gather(assemble_scalar(flux), root=0)
        
        # Valores en dos puntos concretos
        c_lower_val = None
        if len(lower_cell) > 0:
            c_lower_val = c_n.eval(positions_to_evaluate[0], lower_cell[:1])
        c_lower_val = mesh.comm.gather(c_lower_val, root=0)
        
        c_upper_val = None
        if len(upper_cell) > 0:
            c_upper_val = c_n.eval(positions_to_evaluate[1], upper_cell[:1])
        c_upper_val = mesh.comm.gather(c_upper_val, root=0)
        
        if mesh.comm.rank == 0:
            c_flux[i//50] = sum(flux_processor0)
            
            for concentration in c_lower_val:
                if concentration is not None:
                    c_lower[i//50] = concentration[0]
                    break
                
            for concentration in c_upper_val:
                if concentration is not None:
                    c_upper[i//50] = concentration[0]
                    break

# vtx_u.close()
# vtx_c.close()
# vtx_p.close()

if mesh.comm.rank == 0:
    np.save('./plots/c_flux.npy', c_flux)
    np.save('./plots/c_upper.npy', c_upper)
    np.save('./plots/c_lower.npy', c_lower)
