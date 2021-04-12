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
from scipy.optimize import minimize, basinhopping


def data_process(tickers: list, start_day=None, end_day=None):
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
	return returns


def MinVariance(returns=None):
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
		rho = np.array(expected_returns)

	def func(omega):
		return - rho.T.dot(omega) + risk_aversion_coefficient * omega.T.dot(Sigma).dot(omega)

	initial_guess = np.ones(returns.shape()[1]) / returns.shape()[1]
	opt_omega = basinhopping(func, initial_guess, stepsize=5).x

	pass


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
	"""
	pass


def MinTrackingError():
	"""
	最小追踪误差
	"""
	pass


def MaxInformationRatio(expected_active_returns: pd.Series = None, window=252):
	"""
	最大信息比率
	:param expected_active_returns: 预期主动收益率。不传入时，使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益的长度，默认为252，即一年
	"""
	pass


def MaxSharpeRatio(expected_returns: pd.Series = None, window=252):
	"""
	最大化夏普比率
	:param expected_returns: 预期收益率。当不传入时，默认使用历史收益率估计。
	:param window: 使用历史收益率估计预期主动收益时，取历史收益的长度，默认为252，即一年
	"""
	pass


def MaxIndicator():
	pass


def MinStyleDeviation(target_style: pd.Series, relative: bool, priority: pd.Series):
	"""
	:param target_style: 目标风格
	:param relative: 是否为相对于基准
	:param priority: 优先级，可以为每个风格指定一个0-9的优先级，9为最高优先级，0为最低优先级；未指定的风格默认优先级为5
	"""
	pass
