from objective import *


def to_date(date):
	return datetime.datetime.strptime(date, '%Y-%m-%d')


def portfolio_optimize(order_book_ids, date, objective=MinVariance, boundaries: dict = None, cons: list = None,
					   benchmark=None, cov_model=None, factor_risk_aversion=1, specific_risk_aversion=1):
	"""
	:param order_book_ids: 候选合约
	:param date: 优化日期
	:param objective: 目标函数，默认为MinVariance（风险最小化）。
	:param boundaries:个股权重上下界。字典，key 为 order_book_id, value 为 (lower_limit, upper_limit) 组成的 tuple。
					  lower_limit/upper_limit 取值可以是 [0, 1] 的数或 None。当取值为 None 时，表示对应的界不做限制。
					  当 key 为 ‘*’ 时，表示所有未在此字典中明确指定的其他合约。 所有合约默认上下界为 [0, 1]。
	:param cons: 约束列表
	:param benchmark: 基准, 指数使用指数代码，自定义基准需要继承自 OptimizeBenchmark 的类实例
	:param cov_model: 协方差模型，支持 daily/monthly/quarterly
	:param factor_risk_aversion: 因子风险厌恶系数，默认为1
	:param specific_risk_aversion: 特异风险厌恶系数，默认为1
	:return: pd.Series 组合最优化权重
	"""

	returns = data_process(order_book_ids, date)

	bounds = []
	if boundaries is not None:
		for i in range(len(order_book_ids)):
			if order_book_ids[i] in boundaries.keys():
				bounds.append(boundaries[order_book_ids[i]])
			elif '*' in boundaries.keys():
				bounds.append(boundaries[order_book_ids['*']])
			else:
				bounds.append((0, 1))
	else:
		[bounds.append((0, 1)) for i in range(returns.shape[1])]
		cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

	constraints, constraints_hard = dict(), dict()
	if cons is not None:
		for i in range(len(cons)):
			temp = cons[i]
			if len(temp) == 1:
				constraints = dict(constraints, **temp)
			else:
				constraints = dict(constraints, **(temp[0]))
				constraints_hard = dict(constraints_hard, **(temp[1]))
			pass

	if cov_model is None:
		pass
	else:
		pass

	if objective is MinVariance:
		return MinVariance(returns=returns)
	elif objective is MeanVariance:
		return MeanVariance(returns=returns)
	else:
		print('Improper objective.')
		exit(1)

	pass


if __name__ == "__main__":
	pool = ['600969', '300649', '603037', '002111']
	target_date = '2021-04-11'
	portfolio_optimize(pool, target_date, objective=MeanVariance)
