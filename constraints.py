import pandas as pd



def TrackingErrorLimit(upper_limit, hard):
	"""
	追踪误差约束。基准为优化时指定的基准。
	:param upper_limit: 追踪误差上界
	:param hard: 软约束/硬约束
	"""
	constraints = {}

	pass


def TurnoverLimit(current_holding: pd.Series, upper_limit, hard):
	"""
	换手率约束
		Bilateral_turnover_rate = 0.5 * sum(omega[i] - current_holding[i])

	:param current_holding: 当前持仓
	:param upper_limit: 追踪误差上界
	:param hard: 软约束/硬约束
	"""

	pass


def BenchmarkComponentWeightLimit(lower_limit, hard):
	"""
	成分股权重约束，即要求优化结果中，基准成分股的权重之和的下限。基准成分股权重限制。基准为优化时指定的基准。
	:param lower_limit: 基准成分股权重之和下界
	:param hard: 软约束/硬约束
	"""
	pass


def IndustryConstraint(industries, lower_limit, upper_limit, relative, classification, hard):
	"""
	行业权重约束，默认行业分类为申万一级。可选中信一级及申万一级(拆分非银金融行业)
	:param industries: 行业名，可以为单个或多个行业
	:param lower_limit: 行业权重下界。当 relative 为 True 时，支持 ‘-5%’/’+5%’ 形式
	:param upper_limit: 行业权重上界。与下界类似。
	:param relative: 是否为相对于基准
	:param classification: 行业分类标准
	:param hard: 软约束/硬约束
	"""
	pass


def WildcardIndustryConstraint(exclude, lower_limit, upper_limit, relative, classification, hard):
	"""
	指定除 exclude 列表外其他行业的权重限制。与 IndustryConstraint 配合可方便的实现对所有行业权重的限制
	:param exclude: 排除列表
	:param lower_limit: 行业权重下界。当 relative 为 True 时，支持 ‘-5%’/’+5%’ 形式
	:param upper_limit: 行业权重上界。与下界类似。
	:param relative: 是否为相对于基准
	:param classification: 行业分类标准
	:param hard: 软约束/硬约束
	"""
	pass


def StyleConstraint(styles, lower_limit, upper_limit, relative, hard):
	"""
	风格约束
	:param styles : 风格列表，可以为单个或多个风格名。支持以下风格：beta, book_to_price, earnings_yield, growth,
				    leverage, liquidity, momentum, non_linear_size, residual_volatility, size
	:param lower_limit: 行业权重下界。当 relative 为 True 时，支持 ‘-5%’/’+5%’ 形式
	:param upper_limit: 行业权重上界。与下界类似。
	:param relative: 是否为相对于基准
	:param hard: 软约束/硬约束
	"""
	pass


def IndustryComponentLimit(industry, lower_limit, upper_limit, classification, hard):
	"""
	行业内个股权重约束
	:param industry : 行业名
	:param lower_limit: 行业权重下界。当 relative 为 True 时，支持 ‘-5%’/’+5%’ 形式
	:param upper_limit: 行业权重上界。与下界类似。
	:param classification: 行业分类标准
	:param hard: 软约束/硬约束
	"""
	pass


def WildcardStyleConstraint(exclude=None, lower_limit=None, upper_limit=None, relative=False, hard=True):
	"""
	指定除 exclude 外的其他风格上下界，与 WildcardIndustryConstraint 类似。
	:param exclude: 排除列表, 可以为字符串或列表，默认为空，即不排除任何一个风格
	:param lower_limit: 下界。当 relative 为 True 时，支持 ‘-5%’/’+5%’ 形式
	:param upper_limit: 上界。与下界类似。
	:param relative: 是否为相对于基准
	:param hard: 软约束/硬约束
	"""
