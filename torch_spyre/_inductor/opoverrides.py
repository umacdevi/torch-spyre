from torch._inductor.codegen.common import OpOverrides


class SpyreKernelOverrides(OpOverrides):
    """
    Additional ops that are defined for the Spyre device.

    We don't actually use the strings returned from these methods for anything;
    the only thing that is significant is that the method is defined.

    Keep these ops sorted in alphabetical order!
    """

    @staticmethod
    def abs(x):
        return f"spyre.abs({x})"

    @staticmethod
    def exp(x):
        return f"spyre.exp({x})"

    @staticmethod
    def exx2(a, b, c):
        return f"spyre.exx2({a} {b} {c})"

    @staticmethod
    def fma(x):
        return f"spyre.fma({x})"

    @staticmethod
    def layernormnorm(a, b, c, d, e):
        return f"spyre.layernormnorm({a}, {b}, {c}, {d}, {e})"

    @staticmethod
    def layernormscale(x, y):
        return f"spyre.layernormscale({x}, {y})"

    @staticmethod
    def log(x):
        return f"spyre.log({x})"

    @staticmethod
    def reciprocal(x):
        return f"spyre.reciprocal({x})"

    @staticmethod
    def relu(x):
        return f"spyre.relu({x})"

    @staticmethod
    def rsqrt(x):
        return f"spyre.rsqrt({x})"

    @staticmethod
    def sigmoid(x):
        return f"spyre.sigmoid({x})"

    @staticmethod
    def sqrt(x):
        return f"spyre.sqrt({x})"

    @staticmethod
    def tanh(x):
        return f"spyre.tanh({x})"

    @staticmethod
    def where(x, y, z):
        return f"spyre.where({x}, {y}, {z})"


SpyreKernelOverrides._initialize_pointwise_overrides("halide")
