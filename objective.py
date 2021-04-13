# MinVariance: 风险最小化
# MeanVariance: 均值（收益）方差（风险）模型
# RiskParity: 风险平价
# MinTrackingError: 最小追踪误差
# MaxInformationRatio: 最大信息比率
# MaxSharpeRatio: 最大夏普率
# MaxIndicator: 指标值最大化
# MinStyleDeviation: 风格偏离最小化

import numpy as np
import tushare as ts
import pandas as pd
from scipy.optimize import minimize  # , basinhopping
import datetime
from dateutil.relativedelta import relativedelta


def data_process(tickers: list, date, start_day=None, end_day=None):
	if end_day is None:
		end_day = (datetime.datetime.strptime(date, '%Y-%m-%d') - relativedelta(days=1)).strftime('%Y-%m-%d')
	if start_day is None:
		(datetime.datetime.strptime(end_day, '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')

	returns = pd.DataFrame()
	dates = []
	for ticker in tickers:
		try:
			close = ts.get_hist_data(ticker, start_day, end_day)['close']
			dates.append(close.keys())
			prices = np.array(close)
		except ValueError:
			print('Fail to find prices of stock ' + ticker)
		else:
			try:
				returns[ticker] = prices[1:] / prices[:-1] - 1
			except ValueError:
				print('The length of prices we can get for ' + ticker +
					  ' is different from that for previous stocks(s).')
	print(returns)
	return returns


def MinVariance(returns):
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

	return pd.DataFrame(P_star, index=returns.columns, columns=['Suggested weight'])


def MinActiveVariance():
	"""
	主动风险最小化
	"""
	pass


def MeanVariance(returns, expected_returns: pd.Series = None, window=252, risk_aversion_coefficient=0):
	"""
	收益/风险优化
	max mu.T.dot(omega) - lambda * omega.T.dot(Sigma).dot(omega), where lambda stands for risk aversion coefficient

	:param returns: pd.DataFrame, returns of stocks
	:param expected_returns: (None | pd.Series) – 预期收益率。当不传入时，默认使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益率的长度，默认为252，即一年
	:param risk_aversion_coefficient: 风险厌恶系数
	:return:
	"""
	Sigma = returns.cov()
	if expected_returns is None:
		rho = np.mean(returns, axis=0)
	else:
		rho = np.array(expected_returns.fillna(0))

	def fun(omega):
		return - rho.T.dot(omega) + risk_aversion_coefficient * omega.T.dot(Sigma).dot(omega)

	initial_guess = np.ones(returns.shape[1]) / returns.shape[1]

	bounds = []
	[bounds.append((0, 1)) for i in range(returns.shape[1])]
	cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

	opt_omega = minimize(fun, initial_guess, bounds=bounds, constraints=cons).x

	return pd.DataFrame(opt_omega, index=returns.columns, columns=['Suggested weight'])


def ActiveMeanVariance(expected_active_returns: pd.Series = None, window=252, risk_aversion_coefficient=0):
	"""
	主动收益/主动风险优化
	:param expected_active_returns : (None | pd.Series) – 预期主动收益率。当不传入时，默认使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益率的长度，默认为252，即一年
	:param risk_aversion_coefficient: 风险厌恶系数
	"""
	pass


def RiskParity(returns):
	"""
	风险平价
	min sum( ((omega[i] * Sigma.dot(omega)[i] - omega[j] * Sigma.dot(omega)[j])/np.sqrt(omega.T.dot(Sigma).dot(omega))) ** 2 )
	"""

	Sigma = returns.cov()

	def fun(omega):
		intermediate = Sigma.dot(omega)
		denominator = np.sqrt(omega.T.dot(Sigma).dot(omega))
		s = 0
		for i in range(len(omega)):
			for j in range(len(omega)):
				s += ((omega[i] * intermediate[i] - omega[j] * intermediate[j]) / denominator) ** 2
		return s

	initial_guess = np.ones(returns.shape()[1]) / returns.shape()[1]
	bounds = []
	[bounds.append((0, 1)) for i in range(returns.shape[1])]
	cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

	opt_omega = minimize(fun, initial_guess, bounds=bounds, constraints=cons).x

	return opt_omega


def MinTrackingError(returns, baseline_weight):
	"""
	最小追踪误差
	min np.sqrt((omega - baseline_weight).T.dot(Sigma).dot((omega - baseline_weight)))
	"""

	Sigma = returns.cov()

	def fun(omega):
		return np.sqrt((omega - baseline_weight).T.dot(Sigma).dot((omega - baseline_weight)))

	initial_guess = np.ones(returns.shape()[1]) / returns.shape()[1]
	bounds = []
	[bounds.append((0, 1)) for i in range(returns.shape[1])]
	cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

	opt_omega = minimize(fun, initial_guess, bounds=bounds, constraints=cons).x

	return opt_omega


def MaxInformationRatio(returns, expected_active_returns: pd.Series = None, baseline_weight=None, window=252):
	"""
	最大信息比率
	min (weight_p-weight_b).T × (expected_active_returns) / sqrt( (weight_p-weight_b).T × Sigma × (weight_p-weight_b) )

	:param returns: output of data_process function
	:param baseline_weight: 基准组合权重向量
	:param expected_active_returns: 预期主动收益率。不传入时，使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益的长度，默认为252，即一年
	"""

	Sigma = returns.cov()

	def fun(omega):
		return (omega - baseline_weight).T.dot(expected_active_returns) / np.sqrt(
			(omega - baseline_weight).T.dot(Sigma).dot((omega - baseline_weight)))

	initial_guess = np.ones(returns.shape()[1]) / returns.shape()[1]
	bounds = []
	[bounds.append((0, 1)) for i in range(returns.shape[1])]
	cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

	opt_omega = minimize(fun, initial_guess, bounds=bounds, constraints=cons).x

	return opt_omega


def MaxSharpeRatio(returns, expected_returns: pd.Series = None, window=252):
	"""
	最大化夏普比率
	max weight.T × expected_returns / sqrt( weight.T × Sigma × weight )

	:param returns: output of data_process function
	:param expected_returns: 预期收益率。当不传入时，默认使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益的长度，默认为252，即一年
	"""

	Sigma = returns.cov()

	def fun(omega):
		return omega.T.dot(expected_returns) / np.sqrt(omega.T.dot(Sigma).dot(omega))

	initial_guess = np.ones(returns.shape()[1]) / returns.shape()[1]
	bounds = []
	[bounds.append((0, 1)) for i in range(returns.shape[1])]
	cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

	opt_omega = minimize(fun, initial_guess, bounds=bounds, constraints=cons).x

	return opt_omega


def MaxIndicator(factor):
	"""
	指标最大化
	max omega.T.dot(factor)

	:param factor: 用户输入的个股指标值
	:return:
	"""

	def fun(omega):
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

	def fun(omega):
		return (omega.T.dot(target_style) - target_style) ** 2

	pass


"""
风格因子字段 					解释
beta 						个股/投资组合收益对基准组合价格变动的敏感度
momentum 					股票收益变化的总体趋势特征
size 						上市公司的市值规模
earnings_yield 				上市公司的营收能力
residual_volatility 		个股残余收益的波动程度
growth 						上市公司的营收增长情况
book_to_price 				上市公司的股东权益-市值比，反映其估值水平
leverage 					上市公司企业负债占资产比例，反映企业的经营杠杆率
liquidity 					股票换手率，反映个股交易的活跃程度
non_linear_size 			反映中等市值股票和大/小市值股票的表现差异
"""
