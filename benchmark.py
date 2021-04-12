"""
sample:

from rqoptimizer import *

class Benchmark(OptimizeBenchmark):
    def get_weight(self, date):
        return rqdatac.index_weights('000905.XSHG', date)

portfolio_optimize(order_book_ids, date, benchmark=Benchmark(), ...)
"""


def IndexBenchmark(benchmark):
	pass


class OptimizeBenchmark:
	"""
	自定义基准抽象类
	"""

	def get_expected_returns(self, date, window):
		"""
		获得 date 日期的年化期望收益。 仅当使用 MaxInformationRatio 目标函数，且不指定 expected_active_returns 时需要实现此函数
		:param date: datetime.date
		:param window: 回溯窗口大小
		:return: float
		"""

	def get_weight(self, date):
		"""
		获取 date 日期的持仓权重。此函数必需实现。
		:param date: datetime.date
		:return: pd.Series, index 为order_book_id
		"""






