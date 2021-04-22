import datetime
import numpy as np
import tushare as ts
import pandas as pd
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
from opendatatools import swindex
# from jqdata import finance
import jqdatasdk as jq
import time
from itertools import groupby
import cvxopt
import akshare as ak


api = ts.pro_api()

bench = None
returns = pd.DataFrame()


def add_up_to_1(x):
	return sum(x) - 1


constraints, constraints_hard, constraints_des = [{'type': 'eq', 'fun': add_up_to_1}], [True], []
bounds = []
initial_guess = None


def to_date(date):
	return datetime.datetime.strptime(date, '%Y%m%d')


def to_time(date):
	return datetime.datetime.strftime(date, '%Y%m%d')


# -------------------------------------           ------------------------------------ #
# ------------------------------------- Constrain ------------------------------------ #
# -------------------------------------           ------------------------------------ #
category, category_msg = swindex.get_index_list()
content, content_msg = swindex.get_index_cons('801040')


# class TrackingErrorLimit:
#
# 	def __init__(self, upper_limit, hard):
# 		"""
# 		追踪误差约束。基准为优化时指定的基准。
#
# 		:param upper_limit: 追踪误差上界
# 		:param hard: 软约束/硬约束
# 		"""
# 		self.upper_limit = upper_limit
# 		self.hard = hard
# 		self.constraint = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
#
# 	def goal(self, omega):
# 		return


def TrackingErrorLimit(upper_limit, benchmark='sz399552', hard=True):
	"""
	追踪误差约束。基准为优化时指定的基准。

	:param upper_limit: 追踪误差上界
	:param benchmark: 基准（代码格式参照‘sz399552’）
	:param hard: 软约束/硬约束
	"""

	start_d, end_d = returns.index[0], returns.index[-1]
	# start = '2019-01-09'
	try:
		index_df = ak.stock_zh_index_daily(symbol=benchmark)['close']
	except ValueError:
		print('The symbol is invalid.')
		exit(1)
	else:
		try:
			start = np.where(index_df.index == datetime.datetime.strptime(start_d, '%Y-%m-%d'))[0][0]
			end = np.where(index_df.index == datetime.datetime.strptime(end_d, '%Y-%m-%d'))[0][0]
			if end < len(index_df):
				bm = index_df[start:end + 1]
			else:
				bm = index_df[start:]
		except ValueError:
			print('The benchmark you choose is invalid with akshare or does not have enough supported data.')

		def cumulative(lst, t):
			prod = 1
			for k in range(t):
				prod *= lst[t]
			return prod

		def fun(omega):
			TD = []
			for i in range(len(returns.columns)):
				TD.append(cumulative(returns.dot(omega), i) - cumulative(bm, i))
			exp = np.mean(TD)
			TE = np.sqrt(np.sum((TD - exp) ** 2) / (len(TD) - 1))
			return upper_limit - TE

		constraints.append({'type': 'eq', 'fun': fun})
		constraints_hard.append(hard)


def TurnoverLimit(current_holding: pd.Series, upper_limit, hard=True):
	"""
	换手率约束（双边换手率）
		Bilateral_turnover_rate = 0.5 * sum(abs(omega[i] - current_holding[i]))

	:param current_holding: 当前持仓
	:param upper_limit: 追踪误差上界
	:param hard: 软约束/硬约束
	"""

	def func(omega):
		return upper_limit - 0.5 * sum(abs(omega - current_holding))

	constraints.append({'type': 'ineq', 'fun': func})
	constraints_hard.append(hard)


# constraints_des.append({'type': 'ineq', 'name': 'TurnoverLimit', 'hard': hard})


# test = [TurnoverLimit(pd.Series({'1': 1}), 0), TurnoverLimit(pd.Series({'2': 2}), 0)]
# TurnoverLimit.__name__ in [i.__name__ for i in test]


def BenchmarkComponentWeightLimit(lower_limit, hard=True, benchmark='000300'):
	"""
	成分股权重约束，即要求优化结果中，基准成分股的权重之和的下限。基准成分股权重限制。基准为优化时指定的基准。
	:param benchmark: 格式参照‘000300’
	:param lower_limit: 基准成分股权重之和下界
	:param hard: 软约束/硬约束
	"""

	# start = '2019-01-09'
	try:
		index_df = ak.index_stock_cons(index=benchmark)
	except ValueError:
		print('The symbol is invalid.')
		exit(1)
	else:
		lst = index_df['品种代码']

		judge = []

		for k in range(len(returns.columns)):
			if returns.columns[k][:6] in lst:
				judge.append(1)
			else:
				judge.append(0)

		def fun(omega):
			return omega.dot(judge) - lower_limit

		constraints.append({'type': 'eq', 'fun': fun})
		constraints_hard.append(hard)


def IndustryConstraint(industries: list, lower_limit=None, upper_limit=None, relative=False, benchmark='000300',
					   classification=None, hard=True):
	"""
	行业权重约束，默认行业分类为申万一级。可选中信一级及申万一级(拆分非银金融行业)
	:param benchmark:
	:param industries: 行业名，可以为单个或多个行业
	:param lower_limit: 行业权重下界。当 relative 为 True 时，支持 ‘-5%’/’+5%’ 形式
	:param upper_limit: 行业权重上界。与下界类似。
	:param relative: 是否为相对于基准
	:param classification: 行业分类标准
	:param hard: 软约束/硬约束
	"""

	for i in range(len(industries)):

		def func(omega):
			w = 0
			for j in range(len(returns.columns)):
				ind = jq.get_industry(returns.columns[j], date="2021-04-12")
				if industries[i] == ind:
					w += omega[j]
			return w

		if upper_limit is not None:

			if relative:

				tks = [s[:6] for s in returns.columns]
				sw_index_df = ak.sw_index_cons(index_code=benchmark).loc[:, ['stock_code', 'weight']]
				sw_value = [sw_index_df['weight'][k] for k in range(len(sw_index_df)) if sw_index_df['stock_code'][k] in tks]

				def func_u(omega):
					return upper_limit - abs(sum(sw_value) - func(omega))

			else:

				def func_u(omega):
					return upper_limit - func(omega)

			constraints.append({'type': 'ineq', 'fun': func_u})
			constraints_des.append({'type': 'ineq', 'name': 'IndustryConstraint', 'hard': hard})

		if lower_limit is not None:

			if relative:

				def fun_l(omega):
					return abs(sum(sw_value) - func(omega)) - lower_limit

			else:

				def fun_l(omega):
					return func(omega) - lower_limit

			constraints.append({'type': 'ineq', 'fun': fun_l})
			constraints_des.append({'type': 'ineq', 'name': 'IndustryConstraint', 'hard': hard})


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


def reverse_bounds(b):
	rb = []
	for i in b:
		rb.append((-i[1], -i[0]))
	return rb


def time_trans(d):
	temp = time.strptime(d, '%Y-%m-%d')
	out = time.strftime('%Y%m%d', temp)
	return out


def data_process(tickers: list, date, start_day=None, end_day=None):
	if end_day is None:
		end_day = (datetime.datetime.strptime(date, '%Y%m%d') - relativedelta(days=1)).strftime('%Y%m%d')
	if start_day is None:
		start_day = (datetime.datetime.strptime(end_day, '%Y%m%d') - relativedelta(years=1)).strftime('%Y%m%d')

	trading_dates = pd.read_csv('data/trading_dates.csv', encoding='GBK')[:-1].applymap(time_trans)

	d = trading_dates.applymap(to_date)
	range1 = np.where(datetime.datetime.strptime(start_day, '%Y%m%d') < d)[0]
	range2 = np.where(d < datetime.datetime.strptime(end_day, '%Y%m%d'))[0]
	dates = trading_dates.iloc[list(set(range1).intersection(set(range2))), 0]
	r = pd.DataFrame(columns=dates)

	for ticker in tickers[1:]:
		try:
			info = pd.DataFrame(
				ts.pro_bar(ts_code=ticker, adj='qfq', start_date=start_day, end_date=end_day, ma=[5], freq='D'))
		except ValueError:
			print('Fail to find prices of stock ' + ticker)
		else:
			info.index = info['trade_date']
			rtn = info.iloc[[i for i in range(len(info)) if info.index[i] in trading_dates.values.flatten()],
				  :]['pct_chg'] / 100
			rtn.name = ticker
			r = r.append(rtn)

	print('Finish reading stock data.')

	r = r.T
	na_counts = r.isna().sum()
	del_col = []

	for i in range(len(na_counts)):
		if na_counts[i] <= len(dates) / 10:
			na_list = np.where(r.iloc[:, i].isna())

			# group missing values by consecutive dates
			lst = []
			for k, g in groupby(enumerate(na_list), lambda x: x[1] - x[0]):
				lst.append([j for i, j in g])
			lst = [list(item[0]) for item in lst]

			if 0 in lst[0]:
				r.iloc[lst[0][0], i] = r.iloc[lst[0][-1] + 1, i]
				lst = lst[1:]

			if len(dates) in lst[-1]:
				r.iloc[lst[-1], i] = r.iloc[lst[-1][-1] - 1, i]
				lst = lst[:-1]

			if len(lst) > 0 and len(lst[0]) > 0:
				for m in range(len(lst)):
					before = r.iloc[lst[m][0] - 1, i]
					after = r.iloc[lst[m][-1] + 1, i]
					div = (before - after) / len(lst[m])
					fill = [before - div * n for n in range(len(lst[m]))]
					r.iloc[lst[m], i] = fill
		else:
			del_col.append(i)

	r = r.drop(r.columns[del_col], axis=1)

	# sum(r.isna().sum())
	# r = r.applymap(lambda x: 252 * x)

	return r


# -------------------------------------           ------------------------------------ #
# ------------------------------------- Objective ------------------------------------ #
# -------------------------------------           ------------------------------------ #


def MinVariance():
	"""
	风险最小化
	"""
	Sigma = returns.cov()
	Sigma_inv = np.linalg.inv(Sigma)
	rho = np.mean(returns, axis=0)
	e = np.ones(len(rho))

	A = e.dot(Sigma_inv).dot(rho)
	B = rho.dot(Sigma_inv).dot(rho)
	C = e.dot(Sigma_inv).dot(e)
	D = B * C - A ** 2

	g = (B * Sigma_inv.dot(e) - A * Sigma_inv.dot(rho)) / D
	h = (C * Sigma_inv.dot(rho) - A * Sigma_inv.dot(e)) / D

	P_star = g + A / C * h

	constraints.append({'type': 'eq', 'fun': add_up_to_1})

	return pd.DataFrame(P_star, index=returns.columns, columns=['Suggested weight'])


def MinActiveVariance():
	"""
	主动风险最小化
	"""
	pass


def MeanVariance(expected_returns: pd.Series = None, window=252, risk_aversion_coefficient=1):
	"""
	收益/风险优化
	max mu.T.dot(omega) - lambda * omega.T.dot(Sigma).dot(omega), where lambda stands for risk aversion coefficient

	:param expected_returns: (None | pd.Series) – 预期收益率。当不传入时，默认使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益率的长度，默认为252，即一年
	:param risk_aversion_coefficient: 风险厌恶系数
	:return:
	"""
	if window > len(returns):
		window = len(returns)

	Sigma = returns.iloc[-window:, :].cov()
	if expected_returns is None:
		rho = np.mean(returns.iloc[-window:, :], axis=0)
	else:
		rho = np.array(expected_returns.fillna(0))

	def goal(omega):
		return - rho.T.dot(omega) + risk_aversion_coefficient * omega.T.dot(Sigma).dot(omega)

	opt_omega = minimize(goal, initial_guess, bounds=bounds, constraints=constraints).x
	print(-goal(opt_omega))

	return pd.DataFrame(opt_omega, index=returns.columns, columns=['Suggested weight'])


def ActiveMeanVariance(expected_active_returns: pd.Series = None, window=252, risk_aversion_coefficient=0):
	"""
	主动收益/主动风险优化
	:param expected_active_returns : (None | pd.Series) – 预期主动收益率。当不传入时，默认使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益率的长度，默认为252，即一年
	:param risk_aversion_coefficient: 风险厌恶系数
	"""
	pass


def RiskParity():
	"""
	风险平价
	min sum( ((omega[i] * Sigma.dot(omega)[i] - omega[j] * Sigma.dot(omega)[j])/np.sqrt(omega.T.dot(Sigma).dot(omega))) ** 2 )
	"""

	Sigma = returns.cov()

	def goal(omega):
		intermediate = Sigma.dot(omega)
		denominator = np.sqrt(omega.T.dot(Sigma).dot(omega))
		s = 0
		for i in range(len(omega)):
			for j in range(len(omega)):
				s += ((omega[i] * intermediate[i] - omega[j] * intermediate[j]) / denominator) ** 2
		return s

	constraints.append({'type': 'eq', 'fun': add_up_to_1})

	opt_omega = minimize(goal, initial_guess, bounds=bounds, constraints=constraints).x

	return opt_omega


def MinTrackingError(baseline_weight):
	"""
	最小追踪误差
	min np.sqrt((omega - baseline_weight).T.dot(Sigma).dot((omega - baseline_weight)))
	"""

	Sigma = returns.cov()

	def goal(omega):
		return np.sqrt((omega - baseline_weight).T.dot(Sigma).dot((omega - baseline_weight)))

	opt_omega = minimize(goal, initial_guess, bounds=bounds, constraints=constraints).x

	return opt_omega


def MaxInformationRatio(expected_active_returns: pd.Series = None, baseline_weight=None, window=252):
	"""
	最大信息比率
	max (weight_p-weight_b).T × (expected_active_returns) / sqrt( (weight_p-weight_b).T × Sigma × (weight_p-weight_b) )

	:param baseline_weight: 基准组合权重向量
	:param expected_active_returns: 预期主动收益率。不传入时，使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益的长度，默认为252，即一年
	"""

	Sigma = returns.cov()

	def goal(omega):
		return (omega - baseline_weight).T.dot(expected_active_returns) / np.sqrt(
			(omega - baseline_weight).T.dot(Sigma).dot((omega - baseline_weight)))

	opt_omega = minimize(goal, initial_guess, bounds=bounds, constraints=constraints).x

	return opt_omega


def MaxSharpeRatio(expected_returns: pd.Series = None, window=252):
	"""
	最大化夏普比率
	max weight.T × expected_returns / sqrt( weight.T × Sigma × weight )

	:param expected_returns: 预期收益率。当不传入时，默认使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益的长度，默认为252，即一年
	"""

	Sigma = returns.cov()

	def goal(omega):
		return omega.T.dot(expected_returns) / np.sqrt(omega.T.dot(Sigma).dot(omega))

	opt_omega = minimize(goal, initial_guess, bounds=bounds, constraints=constraints).x

	return opt_omega


def MaxIndicator(factor):
	"""
	指标最大化
	max omega.T.dot(factor)

	:param factor: 用户输入的个股指标值
	:return:
	"""

	def goal(omega):
		return omega.T.dot(factor)

	pass


def MinStyleDeviation(target_style: pd.Series, relative: bool, priority: pd.Series):
	"""
	风格偏离最小化
	min (omega.T.(个股因子暴露度) - 目标因子暴露度) ** 2

	:param target_style: 目标风格
	:param relative: 是否为相对于基准
	:param priority: 优先级，可以为每个风格指定一个0-9的优先级，9为最高优先级，0为最低优先级；未指定的风格默认优先级为5
	"""
	base_style = 1

	if relative:
		target_style += base_style

	def goal(omega):
		return (omega.T.dot(target_style) - target_style) ** 2

	opt_omega = minimize(goal, initial_guess, bounds=bounds, constraints=constraints).x

	pass


# -------------------------------------           ------------------------------------ #
# ------------------------------------- Optimizer ------------------------------------ #
# -------------------------------------           ------------------------------------ #


def portfolio_optimize(order_book_ids, date, objective=None, boundaries: dict = None, cons: list = None,
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
	global bench, returns, constraints, constraints_hard, bounds, initial_guess
	bench = benchmark
	returns = data_process(order_book_ids, date)
	initial_guess = np.ones(returns.shape[1]) / returns.shape[1]

	bounds = []
	if boundaries is not None:
		for i in range(returns.shape[1]):
			if order_book_ids[i] in boundaries.keys():
				bounds.append(boundaries[order_book_ids[i]])
			elif '*' in boundaries.keys():
				bounds.append(boundaries[order_book_ids['*']])
			else:
				bounds.append((0, 1))
	else:
		if returns.shape[1] >= 10:
			[bounds.append((0, 3 / returns.shape[1])) for i in range(returns.shape[1])]
		else:
			[bounds.append((0, 1)) for i in range(returns.shape[1])]

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
		return MinVariance()
	elif objective is MeanVariance:
		return MeanVariance()
	else:
		print('Improper objective.')
		exit(1)

	pass


if __name__ == "__main__":
	# pool = ['600969', '300649', '603037', '002111']
	# pool = list(pd.read_csv('data/a.csv').iloc[:, 0])
	target_date = '20210409'

	pool_test = ['000333.XSHE',
				 '000568.XSHE',
				 '000651.XSHE',
				 '000661.XSHE',
				 '000858.XSHE',
				 '002241.XSHE',
				 '002271.XSHE',
				 '002352.XSHE',
				 '002415.XSHE',
				 '002475.XSHE',
				 '002594.XSHE',
				 '002714.XSHE',
				 '300014.XSHE',
				 '300015.XSHE',
				 '300059.XSHE',
				 '300122.XSHE',
				 '300124.XSHE',
				 '300750.XSHE',
				 '300760.XSHE',
				 '300999.XSHE',
				 '600031.XSHG',
				 '600036.XSHG',
				 '600276.XSHG',
				 '600309.XSHG',
				 '600436.XSHG',
				 '600519.XSHG',
				 '600585.XSHG',
				 '600690.XSHG',
				 '600763.XSHG',
				 '600887.XSHG',
				 '600900.XSHG',
				 '601012.XSHG',
				 '601100.XSHG',
				 '601318.XSHG',
				 '601888.XSHG',
				 '603259.XSHG',
				 '603288.XSHG',
				 '603501.XSHG',
				 '603899.XSHG',
				 '688111.XSHG',
				 '688981.XSHG']


	def ticker_transfer(s):
		if s[0] == '3' or s[0] == '0':
			return s[:6] + '.SZ'
		elif s[0] == '6':
			return s[:6] + '.SH'


	pool_test = [ticker_transfer(s) for s in pool_test]

	# data = pd.read_csv('data/a.csv')
	# close = jq.get_price('600969.XSHG')['close']
	# returns = data_process(pool, target_date)

	result = portfolio_optimize(pool_test, target_date, objective=MeanVariance)
	prt = round(result, 4)
	print(prt[prt['Suggested weight'] != 0])
	# returns_mark1 = returns

	"""
	Sample:
	
	#优化日期
	date = '2019-02-28'
	#候选合约
	pool =  index_components('000906.XSHG',date)

	#对组合中所有个股头寸添加相同的约束0~5%
	bounds = {'*': (0, 0.05)}
	#优化函数，风格偏离最小化，优化组合beta风格因子暴露度相对于基准组合高出1个标准差
	objective = MinStyleDeviation({'size': 0, 'beta': 1, 'book_to_price': 0, 'earnings_yield': 0, 'growth': 0, 'leverage': 0, 'liquidity': 0, 'momentum': 0, 'non_linear_size': 0, 'residual_volatility': 0}, relative=True)
	#约束条件，对所有风格因子添加基准中性约束，允许偏离±0.3个标准差
	cons = [
			WildcardStyleConstraint(lower_limit=-0.3, upper_limit=0.3, relative=True, hard=True)
		]

	portfolio_optimize(pool, date, bnds=bounds, cons=cons, benchmark='000300.XSHG',objective = objective)
	"""
	# sum(result.iloc[:, 0])

	f = open("test.txt")
	line = f.readline()
	text_data = ''
	while line:
		text_data += line
		line = f.readline()
	f.close()

	text_data = text_data.split('\n')
	text_ticker, text_weight = [], []
	for x in range(len(text_data)):
		text_ticker.append(text_data[x].split('    ')[0])
		text_weight.append(text_data[x].split('    ')[1])

	text_weight = [float(k) for k in text_weight]

	rq_opt = pd.DataFrame(data=text_weight, index=pool_test)

	dp = []
	for k in range(len(rq_opt)):
		if rq_opt.index[k] not in returns.columns:
			dp.append(k)

	rq_opt = rq_opt.drop(rq_opt.index[dp])

# 0.004349368348357958
