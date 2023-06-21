import numpy as np
import gmsh


# =============================================================================
#                        DIMENSIONES DEL PROBLEMA
#
#       -----------------------------------------------------------------     ^
#                                                                             |
#                                                                             |
#                                                                             |
# ^     -----------------                               -----------------     H
# |                      |                             |                      |
# h                      |                             |                      |
# |                      |                             |                      |
# v                       -----------------------------                       v
#      < - - - l  - - - ><------------- l0 -------------><-----l-------->
#
#       <   -    -    -    -    -    -    L    -    -    -    -    -    ->
# =============================================================================


gdim, fdim = 2, 1
L, l0, l, H, h = 2.0, 1.0, 0.5, 0.8, 0.4
res = h / 16.0

fluid_tag, wall_tag, inlet_tag, outlet_tag, free_tag = 1, 2, 3, 4, 5
fluid, walls, inlet, outlet = [], [], [], []

gmsh.initialize()

channel = gmsh.model.occ.addRectangle(0, h, 0, L, H-h)
h_surf = [
   surf for dim, surf in gmsh.model.occ.getEntities(fdim) if \
       np.allclose(gmsh.model.occ.getCenterOfMass(dim, surf), [1, h, 0])
][0]
gmsh.model.occ.extrude([(fdim, h_surf)], 0, -h, 0)
gmsh.model.occ.synchronize()

lower_vol = [
    vol for dim, vol in gmsh.model.occ.getEntities(gdim) if \
    np.allclose(gmsh.model.occ.getCenterOfMass(dim, vol), [L/2, h/2, 0])
][0]

step1 = gmsh.model.occ.addRectangle(0, 0, 0, l, h)
step2 = gmsh.model.occ.addRectangle(l0+l, 0, 0, l0+0.5, h)
cut = gmsh.model.occ.cut([(gdim, lower_vol)], [(gdim, step1), (gdim, step2)])
gmsh.model.occ.synchronize()

gmsh.model.occ.fragment(
    [(gdim, gmsh.model.occ.getEntities(gdim)[0][1])], 
    [(gdim, gmsh.model.occ.getEntities(gdim)[1][1])], 
)
gmsh.model.occ.synchronize()

fluid.append(gmsh.model.occ.getEntities(gdim)[0][1])
fluid.append(gmsh.model.occ.getEntities(gdim)[1][1])
gmsh.model.addPhysicalGroup(gdim, fluid, fluid_tag)
gmsh.model.setPhysicalName(gdim, fluid_tag, "Fluid")

# # # # Fragmentar el lid
# lid_surfs = [
#     surf for dim, surf in gmsh.model.occ.getEntities(fdim) if \
#     np.allclose(gmsh.model.occ.getCenterOfMass(dim, surf), [L/2, h, 0])
# ]
# gmsh.model.occ.fragment([(fdim, lid_surfs[0])], [(fdim, lid_surfs[1])])
# gmsh.model.occ.synchronize()

# =============================================================================
#                       CONTORNOS Y GRUPOS FÍSICOS
# =============================================================================

surfaces = gmsh.model.occ.getEntities(fdim)

for dim, tag in surfaces:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    
    if np.allclose(com, [0, (H+h)/2, 0]):
        inlet.append(tag)
        
    elif np.allclose(com, [L, (H+h)/2, 0]):
        outlet.append(tag)
        
    elif np.allclose(com, [L/2, H, 0]):
        free = [tag]
    
    elif np.allclose(com, [L/2, h, 0]):
        lid = [tag]
        
    else:
        walls.append(tag)

gmsh.model.addPhysicalGroup(fdim, walls, wall_tag)
gmsh.model.setPhysicalName(fdim, wall_tag, "Walls")
gmsh.model.addPhysicalGroup(fdim, inlet, inlet_tag)
gmsh.model.setPhysicalName(fdim, inlet_tag, "Inlet")
gmsh.model.addPhysicalGroup(fdim, outlet, outlet_tag)
gmsh.model.setPhysicalName(fdim, outlet_tag, "Outlet")
gmsh.model.addPhysicalGroup(fdim, free, free_tag)
gmsh.model.setPhysicalName(fdim, free_tag, "free")
gmsh.model.addPhysicalGroup(fdim, lid, 6)
gmsh.model.setPhysicalName(fdim, 6, "lid")


# =============================================================================
#                       CREAR MALLADO ADAPTATIVO
#   La zona de mayor interés es el cambio de profundidad del canal, por lo que
#  es ahí donde se coloca la mayor resolución.
# =============================================================================

main_surface = [
    surf[1] for surf in gmsh.model.occ.getEntities(fdim) if \
    np.allclose(
        gmsh.model.occ.getCenterOfMass(fdim, surf[1]), [l, h/2, 0]
    )
    or \
    np.allclose(
        gmsh.model.occ.getCenterOfMass(fdim, surf[1]), [l0+l, h/2, 0]
    )
] + inlet

distance_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field().setNumbers(distance_field, "EdgesList", main_surface)

threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", res*1.5)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", h/2)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", h)
min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

# gmsh.option.setNumber("Mesh.Algorithm", 1)
# gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
# gmsh.option.setNumber("Mesh.RecombineAll", 2)
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
gmsh.model.mesh.generate(gdim)
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("Netgen")

gmsh.write('./domains/stepv3.msh')
gmsh.finalize()
