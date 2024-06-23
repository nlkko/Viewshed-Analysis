import math
import sys

from direct.filter.FilterManager import FilterManager
from direct.showbase.ShowBase import ShowBase
from src.sample import Sampler
from panda3d.core import loadPrcFile, Shader, Texture, TransparencyAttrib, LVector4
from src.terrain_mesh import create_mesh
from src.camera_controls import CameraControls
from src.viewshed_analysis.line_of_sight import RayHandler
from src.viewshed_analysis.reference_planes import ReferencePlanes

# Load configs
loadPrcFile(r'config/panda.prc')


class Application(ShowBase):
    def __init__(self, viewshed, sample_data, terrain_mesh, visible_terrain_mesh, chunk, resolution):
        ShowBase.__init__(self)

        self.height = 40
        self.camera_controls = None
        self.viewpoint = None
        self.viewshed = viewshed
        self.sample_data = sample_data
        self.terrain = terrain_mesh
        self.visible_terrain = visible_terrain_mesh
        self.chunk = chunk
        self.resolution = resolution

        self.camLens.setNear(50)

        # self.setBackgroundColor(0.96, 0.38, 0, 1)

        # Fast exit
        self.accept("escape", sys.exit)

        # Scene
        self.setup_scene()

        # Post Processing
        self.setup_post_processing()

    def setup_scene(self):

        # viewpoint represented as a red ball
        ball = self.loader.loadModel("rsc/assets/sphere")
        ball.set_color(LVector4(1, 0, 0, 1))
        ball.set_pos(0, 0, 0)
        ball.set_scale(self.chunk * 0.02)
        ball.set_bin("fixed", 0)
        ball.set_depth_test(False)
        ball.set_depth_write(False)

        self.viewpoint = self.render.attachNewNode(ball.node())


        # Set up task to update camera
        self.camera_controls = CameraControls(self, self.height, self.chunk, self.resolution, self.viewpoint)
        self.taskMgr.add(self.camera_controls.transform_view, 'Rotate Camera View')

        centre = int(math.floor(chunk / 2))
        centre_height = self.sample_data[centre, centre]

        terrain_np = self.render.attachNewNode(self.terrain)
        terrain_np.setPos(0, 0, -centre_height)

        visible_terrain_np = self.render.attachNewNode(self.visible_terrain)
        visible_terrain_np.setPos(0, 0, -centre_height + 5)
        visible_terrain_np.setTransparency(TransparencyAttrib.MAlpha)

        visible_terrain_np.setDepthWrite(False)
        visible_terrain_np.setDepthTest(False)
        visible_terrain_np.setBin('transparent', 1)



    def setup_post_processing(self):
        self.manager = FilterManager(self.win, self.cam)

        color_texture = Texture()
        depth_texture = Texture()

        quad = self.manager.renderSceneInto(colortex=color_texture, depthtex=depth_texture)
        quad.setShader(
            Shader.load(Shader.SL_GLSL, "rsc/shaders/outline/outline.vert", "rsc/shaders/outline/outline.frag"))
        quad.setShaderInput("tex", color_texture)
        quad.setShaderInput("depth_tex", depth_texture)
        quad.setShaderInput("screen_resolution", (self.win.getSize().x, self.win.getSize().y))
        quad.setShaderInput("inv_proj_mat", self.camLens.getProjectionMatInv())

        # anti-aliasing
        # self.render.setAntialias(AntialiasAttrib.MLine)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # viewshed analysis
    lat, lon = 42.209268212033386, 20.955407037566157
    chunk = 400  # size of chunk

    # Sample
    sampler = Sampler(r'data/EU_DTM_be.vrt')
    sample_data = sampler.sample(lat, lon, chunk)

    algorithms = ["reference_planes", "r2", "r3", "r2_ridge", "r3_ridge"]
    algorithm = algorithms[1]

    match algorithm:
        case "reference_planes":
            viewshed_analysis = ReferencePlanes(sample_data)
        case "r2":
            viewshed_analysis = RayHandler(sample_data, sampler.get_gt())
            viewshed_analysis.start_handler_r2()
        case "r3":
            viewshed_analysis = RayHandler(sample_data, sampler.get_gt())
            viewshed_analysis.start_handler_r3()

    viewshed = viewshed_analysis.get_viewshed()[:chunk, :chunk]

    # Create meshes
    terrain, visible_terrain = create_mesh(sample_data, sampler.get_gt(), viewshed, delaunay=False)

    # Plot
    plt.imshow(sample_data, cmap='terrain')
    plt.title("Viewshed Analysis")

    overlay = np.zeros((chunk, chunk, 4))
    overlay[viewshed == 1] = [0, 0, 0, 0.5]
    plt.imshow(overlay, cmap='Reds')

    plt.show()

    app = Application(viewshed, sample_data, terrain, visible_terrain, chunk, sampler.get_gt()[1])
    app.run()
