import math

from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from panda3d.core import Point2, Vec3


# TODO: Add LERP

def resize_viewpoint(viewpoint, cam_pos):
    origin = viewpoint.getPos()
    scale_factor = 2

    distance = (cam_pos - origin).length()

    if distance > 0:
        log_distance = math.log(distance + 1)
        scale = log_distance * scale_factor
    else:
        scale = 0

    viewpoint.set_scale(scale)


class CameraControls:
    def __init__(self, base, height, chunk, resolution, viewpoint):
        self.base = base

        self.mode = "pov"

        self.chunk = chunk
        self.resolution = resolution
        self.height = height

        self.viewpoint = viewpoint

        # disable default camera controls
        self.base.disableMouse()

        self.base.cam.setZ(height)

        self.base.camLens.setFov(60)

        self.cam_pivot = self.base.render.attachNewNode('cam_pivot')
        self.base.cam.reparentTo(self.cam_pivot)

        self.mp_pos = None

        self.m1_flag = False
        self.shift_flag = False
        self.space_flag = False

        self.yaw_sensitivity = 0.5
        self.pitch_sensitivity = 0.1

        self.speed = 100

        # mouse: left click
        self.base.accept("mouse1", self.mouse_down)
        self.base.accept("mouse1-up", self.mouse_up)

        # keyboard: shift
        self.base.accept("shift", self.shift_down)
        self.base.accept("shift-up", self.shift_up)

        # keyboard: shift
        self.base.accept("space", self.space_down)
        self.base.accept("space-up", self.space_up)

        # keyboard: F1 to switch modes
        self.base.accept("f1", self.switch_mode)

    def mouse_down(self):
        self.m1_flag = True
        self.mp_pos = None

    def mouse_up(self):
        self.m1_flag = False
        self.mp_pos = None

    def shift_down(self):
        self.shift_flag = True

    def shift_up(self):
        self.shift_flag = False

    def space_down(self):
        self.space_flag = True

    def space_up(self):
        self.space_flag = False

    def switch_mode(self):
        # reset other cam position
        self.base.cam.setY(0)
        self.base.cam.setZ(0)

        if self.mode == "pov":
            self.mode = "bird"

            self.speed = 1000
            self.yaw_sensitivity = 0.5
            self.pitch_sensitivity = 0.3

            self.cam_pivot.setHpr(0, -45, 0)
            self.base.cam.setY(-(self.chunk * self.resolution + 1000))
        else:
            self.mode = "pov"

            self.speed = 100
            self.yaw_sensitivity = 0.5
            self.pitch_sensitivity = 0.1

            self.cam_pivot.setHpr(0, 0, 0)
            self.base.cam.setZ(self.height)

    def transform_view(self, task):
        if self.m1_flag and self.base.mouseWatcherNode.getMouse():  # if m1 pressed & mouse inside window
            mc_pos = self.base.mouseWatcherNode.getMouse()  # current mouse position
            if self.mp_pos is None:
                self.mp_pos = Point2(mc_pos)
            else:
                d_pitch = (mc_pos.y - self.mp_pos.y) * 100.0 * self.pitch_sensitivity
                d_yaw = (mc_pos.x - self.mp_pos.x) * 100.0 * self.yaw_sensitivity
                pivot = self.cam_pivot
                pivot.set_hpr(pivot.get_h() + d_yaw, pivot.get_p() - d_pitch, 0.0)
                self.mp_pos = Point2(mc_pos)

        dt = globalClock.getDt()  # delta time

        if self.mode == 'pov':
            if self.shift_flag:
                # move the camera up
                self.cam_pivot.setPos(self.cam_pivot.getPos() - Vec3(0, 0, self.speed * dt))

            if self.space_flag:
                # move the camera down
                self.cam_pivot.setPos(self.cam_pivot.getPos() + Vec3(0, 0, self.speed * dt))
        else:
            if self.shift_flag:
                # move the camera closer to the viewpoint
                self.base.cam.setY(self.base.cam, -self.speed * dt)

            if self.space_flag:
                # move the camera farther from the viewpoint
                self.base.cam.setY(self.base.cam, self.speed * dt)

        resize_viewpoint(self.viewpoint, self.base.cam.getPos())

        return Task.cont
