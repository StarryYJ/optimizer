


if __name__ == "__main__":
	"""
	# 优化日期
	date = '2019-02-28'
	# 候选合约
	pool = index_components('000906.XSHG', date)

	# 对组合中所有个股头寸添加相同的约束0~5%
	bounds = {'*': (0, 0.05)}
	# 优化函数，风格偏离最小化，优化组合beta风格因子暴露度相对于基准组合高出1个标准差
	objective = MinStyleDeviation(
		{'size': 0, 'beta': 1, 'book_to_price': 0, 'earnings_yield': 0, 'growth': 0, 'leverage': 0, 'liquidity': 0,
		 'momentum': 0, 'non_linear_size': 0, 'residual_volatility': 0}, relative=True)
	# 约束条件，对所有风格因子添加基准中性约束，允许偏离±0.3个标准差
	cons = [
		WildcardStyleConstraint(lower_limit=-0.3, upper_limit=0.3, relative=True, hard=True)
	]

	portfolio_optimize(pool, date, bnds=bounds, cons=cons, benchmark='000300.XSHG', objective=objective)
	"""

