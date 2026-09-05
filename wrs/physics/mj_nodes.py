import wrs.physics.mj_compiler as mjc


class WorldNode:
    def __init__(self):
        self.name = "world"
        self.option = None
        self.default = None
        self.assets = []
        self.root_body = None
        self.sensors = []
        self.actuators = []
        self.contact_excludes = []
        self._compiler = mjc.MJCFCompiler()

    def compile_mjcf(self):
        return self._compiler.compile_mjcf(self)


class DefaultNode:
    def __init__(self):
        self.joint = {}
        self.geom = {}
        self.motor = {}


class OptionNode:
    def __init__(self):
        self.gravity = (0, 0, -9.81)
        self.timestep = 0.002
        self.integrator = "Euler"  # 或 RK4
        self.solver = "Newton"


class AssetNode:
    def __init__(self, name):
        self.name = name


class MeshAsset(AssetNode):
    def __init__(self, name, path=None, vertices=None):
        super().__init__(name)
        # Either a mesh FILE path, or an inline VERTEX list (flat x y z ...). With
        # inline vertices MuJoCo builds the mesh's convex hull from the points --
        # used for collision proxies whose source STL MuJoCo cannot load (e.g. a
        # huge ASCII mesh), where we pass a precomputed convex hull's vertices.
        self.path = path
        self.vertices = vertices


# class TextureAsset(AssetNode):
#     def __init__(self, name, path):
#         super().__init__(name)
#         self.path = path


# class MaterialAsset(AssetNode):
#     def __init__(self, name, rgba=None):
#         super().__init__(name)
#         self.rgba = rgba


class BodyNode:
    def __init__(self, name):
        self.name = name
        self.pos = (0, 0, 0)
        self.quat = (0, 0, 0, 1)
        self.inertial = None
        self.geoms = []
        self.sites = []
        self.hosting_jnts = []
        self.children = []
        self.parent = None


class JointNode:
    def __init__(self, name: str):
        self.name = name
        self.jtype_str = "hinge"  # hinge / slide / fixed
        self.ax = (1, 0, 0)
        # self.pos = (0, 0, 0)
        # self.quat = (0, 0, 0, 1)
        self.range = None
        self.damping = 1
        self.frictionloss = 0.01
        self.armature = .02


class InertialNode:
    def __init__(self, mass, com=(0, 0, 0),
                 inertia=None):
        self.mass = mass
        self.com = com
        self.inertia = inertia


class GeomNode:
    def __init__(self, name: str):
        self.name = name
        self.gtype = "box"  # sphere, capsule, mesh...
        self.size = (1, 1, 1)
        self.pos = (0, 0, 0)
        self.quat = (0, 0, 0, 1)
        self.rgba = None
        self.mesh_ref = None
        self.friction = (2, 0.1, 0.01)
        self.contype = None
        self.conaffinity = None


class SiteNode:
    """A massless, non-colliding marker frame rigidly attached to a body.

    This is what a collision-free mounted SceneObject (a tcp coordinate frame, a
    marker) becomes: MuJoCo computes its world pose every step -- readable as
    ``data.site_xpos`` / ``site_xmat`` -- but it contributes NO mass, NO contact
    and NO degrees of freedom, and it lives outside the body tree, so it can
    never perturb its host link the way a zero-geom child body would. Sites take
    primitive shapes only (no mesh), so the marker is a small sphere by default;
    the viewer draws the frame axes from the site's own pose."""

    def __init__(self, name: str):
        self.name = name
        self.stype = "sphere"   # sphere / capsule / ellipsoid / cylinder / box
        self.size = (0.005,)
        self.pos = (0, 0, 0)
        self.quat = (0, 0, 0, 1)
        self.rgba = None


class ActuatorNode:
    def __init__(self, name):
        self.name = name
        self.atype = "position"  # position/velocity/motor
        self.joint = None
        self.kp = 200.0
        self.kv = None

# class SensorNode:
#     def __init__(self, name: str):
#         self.name = name
#         self.stype = "jointpos"  # jointpos/jointvel/force/accel...
#         self.source_joint: JointNode | None = None
#         self.source_body: BodyNode | None = None


# class RobotNode:
#     def __init__(self, name: str):
#         self.name = name
#         self.root_body = None
#         self.joints = []
#         self.actuators = []
