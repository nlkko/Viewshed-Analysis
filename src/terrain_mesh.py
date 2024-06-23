import numpy
import matplotlib.pyplot as plt
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode
from scipy.spatial import Delaunay

# TODO:
# * Calculate normals
# * For optimization add LOD, if too many points
# * Calculate the color with shaders, instead of color data

def create_mesh(sample_data, geotransform, viewshed, delaunay = False):

    points = []

    for j in range(len(sample_data[0])):
        for i in range(len(sample_data)):
            points.append([i, j])

    # set up panda3d mesh
    format = GeomVertexFormat.getV3c4t2()  # vertex format - position, color, texture coordinates

    # terrain mesh
    vertex_data = GeomVertexData('terrain', format, Geom.UHStatic)
    vertex = GeomVertexWriter(vertex_data, 'vertex')
    color = GeomVertexWriter(vertex_data, 'color')
    texcoord = GeomVertexWriter(vertex_data, 'texcoord')

    # viewshed mesh
    visible_vertex_data = GeomVertexData('visible_terrain', format, Geom.UHStatic)
    visible_vertex = GeomVertexWriter(visible_vertex_data, 'vertex')
    visible_color = GeomVertexWriter(visible_vertex_data, 'color')
    visible_texcoord = GeomVertexWriter(visible_vertex_data, 'texcoord')

    # get correct resolution from geotransform
    ps_x = geotransform[1]  # pixel size x-direction
    ps_y = -geotransform[5]  # pixel size y-direction, negative

    # get centre of mesh in order to centre mesh on local coordinate system - float is allowed here
    centre = numpy.mean(points, axis=0)

    # normalize z values in order that colors gradient don't become uniform
    z_values = numpy.array(sample_data).flatten()
    normalized_z_values = (z_values - min(z_values)) / (max(z_values) - min(z_values))

    for id, (i, j) in enumerate(points):
        # centre mesh by subtracting centre at given position
        centre_i = (i - centre[0]) * ps_x
        centre_j = (j - centre[1]) * -ps_y

        vertex.addData3f(centre_i, centre_j, z_values[id])
        visible_vertex.addData3f(centre_i, centre_j, z_values[id])

        # add color based on matplotlib colormap
        color_value = plt.cm.terrain(normalized_z_values[id])
        color.addData4f(color_value[0], color_value[1], color_value[2], color_value[3])

        # add color for visible terrain
        visible_color.addData4f(0, 0, 0, 0.5)

        # add texcoord
        texcoord.addData2f(i, j)
        visible_texcoord.addData2f(i, j)

    visible_points = numpy.array(viewshed).flatten()

    # tris
    tris = GeomTriangles(Geom.UHStatic)
    visibility_tris = GeomTriangles(Geom.UHStatic)

    # add triangles
    if delaunay:
        triangulation = Delaunay(points)

        for triangle in triangulation.simplices:

            tris.addVertices(triangle[0], triangle[2], triangle[1])

            if any(visible_points[triangle]):
                visibility_tris.addVertices(triangle[0], triangle[2], triangle[1])

    else:
        for j in range(len(sample_data) - 1):
            for i in range(len(sample_data[0]) - 1):

                # list vertices to create 2 triangles in a rectangle - using index

                v1 = j * len(sample_data) + i
                v2 = (j + 1) * len(sample_data) + i
                v3 = j * len(sample_data) + (i + 1)
                v4 = (j + 1) * len(sample_data) + (i + 1)

                tris.addVertices(v1, v2, v3)
                tris.addVertices(v3, v2, v4)

                if visible_points[v3] or visible_points[v2] or visible_points[v1]: visibility_tris.addVertices(v1, v2, v3)
                if visible_points[v4] or visible_points[v2] or visible_points[v3]: visibility_tris.addVertices(v3, v2, v4)

    # add to panda3d node
    geom = Geom(vertex_data)
    geom.addPrimitive(tris)
    node = GeomNode('terrain')
    node.addGeom(geom)

    visibility_geom = Geom(visible_vertex_data)
    visibility_geom.addPrimitive(visibility_tris)
    visibility_node = GeomNode('visible_terrain')
    visibility_node.addGeom(visibility_geom)

    return node, visibility_node
