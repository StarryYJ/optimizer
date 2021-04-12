
class OptimizerException(Exception):
    pass


class OptimizerWarning(Warning):
    pass


class InvalidArgument(OptimizerException):
    """参数错误"""
    pass


class OptimizationFailed(OptimizerException):
    """优化失败"""
    pass



