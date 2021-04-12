"""
CovModel 协方差模型:
FACTOR_MODEL_DAILY 日度协方差模型
FACTOR_MODEL_MONTHLY 月度协方差模型
FACTOR_MODEL_QUARTERLY 季度协方差模型

IndustryClassification 行业分类标准:
SWS 申万一级
ZX 中信一级
SWS_1 申万一级，对非银金融进一步细分

风格因子：
beta 贝塔
book_to_price 账面市值比
earnings_yield 盈利率
growth 成长性
leverage 杠杆率
liquidity 流动性
momentum 动量
non_linear_size 非线性市值
residual_volatility 残余波动率
size 市值

"""

from enum import Enum


class CovModel(Enum):
	FACTOR_MODEL_DAILY = 'factor_model/daily'
	FACTOR_MODEL_MONTHLY = 'factor_model/monthly'
	FACTOR_MODEL_QUARTERLY = 'factor_model/quarterly'


class IndustryClassification(Enum):
	"""行业分类标准"""
	SWS = 'shenwan'  # 申万一级
	ZX = 'zhongxin'  # 中信一级
	SWS_1 = 'shenwan_non_bank_financial_breakdown'  # 申万一级，对非银金融进一步细分


STYLE_FACTORS = [
	'beta', 'book_to_price', 'earnings_yield', 'growth', 'leverage', 'liquidity', 'momentum',
	'non_linear_size', 'residual_volatility', 'size'
]
