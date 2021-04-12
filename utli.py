
from collections import namedtuple
import warnings
import numpy as np
import pandas as pd

import rqdatac
from .helper import OptimizeHelper
from const import CovModel
from exception import OptimizerWarning, InvalidArgument
from .optimize import to_date
from .benchmark import OptimizeBenchmark, IndexBenchmark

"""总股数目标"""
TotalStockCountTarget = namedtuple('TotalStockCountTarget', 'count')
"""行业股票数目"""
IndustryStockCountTarget = namedtuple('IndustryStockCountTarget', 'industry,classification,count')


class RiskScore:
    """
    风险评分。风险越高，分数越低
    """
    def get_score(self, helper):
        order_book_ids = helper.order_book_ids
        weight = np.repeat(1.0 / len(order_book_ids), len(order_book_ids))
        risk = weight.dot(helper.get_cov_matrix())
        return pd.Series(1.0 / risk, index=order_book_ids)


class ActiveRiskScore:
    """
    主动风险评分。主动风险越高，分数越低
    """
    def get_score(self, helper):
        order_book_ids = helper.order_book_ids
        weight = np.repeat(1.0 / len(order_book_ids), len(order_book_ids))
        not_in_pool = len(helper.union_components) - len(helper.order_book_ids)
        if not_in_pool > 0:
            active_weight = np.hstack([weight, np.repeat(0.0, not_in_pool)]) - helper.benchmark_weight.values
        else:
            active_weight = weight - helper.benchmark_weight.values

        active_risk = active_weight.dot(helper.get_union_cov_matrix())
        return pd.Series(1.0 / active_risk[:len(order_book_ids)], index=order_book_ids)


class RiskReturnScore:
    """
    风险收益打分。收益 / 风险，越高分数越高
    :param expected_returns: 预期收益。None时使用历史收益估计
    :param window: 历史收益估计的窗口大小，默认为252
    """
    def __init__(self, expected_returns=None, window=252):
        self.expected_returns = expected_returns
        self.window = window

    def get_score(self, helper):
        order_book_ids = helper.order_book_ids
        weight = np.repeat(1.0 / len(order_book_ids), len(order_book_ids))
        risk = weight.dot(helper.get_cov_matrix())
        rtn = self.expected_returns
        if rtn is None:
            start_date = rqdatac.trading_date_offset(helper.prev_trading_date, -self.window)
            rtn = rqdatac.get_price_change_rate(order_book_ids, start_date, helper.prev_trading_date).mean() * 252
        rtn = rtn.reindex(order_book_ids, fill_value=0.0)
        return rtn / risk


class ActiveRiskReturnScore:
    """
    主动风险收益打分。主动收益 / 主动风险，越高分数越高
    :param expected_active_returns: 预期主动收益。None时使用历史收益估计
    :param window: 历史收益估计的窗口大小，默认为252
    """
    def __init__(self, expected_active_returns=None, window=252):
        self._expected_active_returns = expected_active_returns
        self.window = window

    def get_score(self, helper):
        order_book_ids = helper.order_book_ids
        weight = np.repeat(1.0 / len(order_book_ids), len(order_book_ids))
        not_in_pool = len(helper.union_components) - len(helper.order_book_ids)
        if not_in_pool > 0:
            active_weight = np.hstack([weight, np.repeat(0.0, not_in_pool)]) - helper.benchmark_weight.values
        else:
            active_weight = weight - helper.benchmark_weight.values
        active_risk = active_weight.dot(helper.get_union_cov_matrix())

        rtn = self._expected_active_returns
        if rtn is None:
            start_date = rqdatac.trading_date_offset(helper.prev_trading_date, -self.window)
            rtn = rqdatac.get_price_change_rate(order_book_ids + [helper.benchmark], start_date,
                                                                  helper.prev_trading_date).mean() * 252
            benchmark_returns = rtn[helper.benchmark]
            rtn -= benchmark_returns

        rtn = rtn.reindex(order_book_ids, fill_value=0.0)
        return rtn / active_risk[:len(order_book_ids)]


class TargetStyleScore:
    """
    目标风格打分。与目标风格越接近分数越高
    :param target_style: pd.Series
    :param relative: 是否相对于基准
    """
    def __init__(self, target_style, relative=False):
        self.target_style = target_style
        self.relative = relative

    def get_score(self, helper):
        target_style = self.target_style
        if self.relative:
            benchmark_style = helper.get_benchmark_style()
            target_style += benchmark_style

        style = helper.get_factor_exposure()
        score = style.subtract(target_style).abs().sum(axis=1)
        return -score


def stock_select(pool, targets, date, score, benchmark=None):
    """
    选股
    :param pool: pd.Series, index 为order_book_id, value 为 0/1/2，其中0表示最高优先级股票池，1表示次优先选择股票池，2表示备选股票池
    :param targets: [TotalStockCountTarget | IndustryStockCountTarget]
    :param date: 选股日期
    :param score: RiskScore | RiskReturnScore | TargetStyleScore
    :param benchmark: 基准
    :return: [order_book_id]
    """
    selected = []
    pool_0 = pool[pool == 0].index.tolist()
    pool_1 = pool[pool == 1].index.tolist()
    pool_2 = pool[pool == 2].index.tolist()

    date = to_date(date)

    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark = IndexBenchmark(benchmark)
        if not isinstance(benchmark, OptimizeBenchmark):
            raise InvalidArgument('无效的benchmark：{}；对于指数，可以直接传入对应的order_book_id；'
                                  '对于自定义基准，请传入一个 OptimizeBenchmark 的对象'.format(benchmark))

    helper = OptimizeHelper(pool_0 + pool_1 + pool_2, date, benchmark, CovModel.FACTOR_MODEL_DAILY)
    if not isinstance(score, pd.Series):
        score = score.get_score(helper)

    pools = [
        score[pool_0].sort_values(ascending=False),
        score[pool_1].sort_values(ascending=False),
        score[pool_2].sort_values(ascending=False)
    ]

    industry_stock_count_targets = [t for t in targets if isinstance(t, IndustryStockCountTarget)]

    for t in industry_stock_count_targets:
        mapping = helper.get_industry_mapping(t.classification)
        candidates = set(mapping[mapping == t.industry].index)

        need = t.count
        for pool in pools:
            pool_candidates = [s for s in pool.index if s in candidates]
            selected += pool_candidates[:need]
            need -= len(pool_candidates)
            for s in pool_candidates:
                del pool[s]
            if need <= 0:
                break

        if need > 0:
            warnings.warn('股票池中行业 {} 股票数目不足'.format(t.industry), category=OptimizerWarning)

    total_stock_count_target = [t for t in targets if isinstance(t, TotalStockCountTarget)]
    if len(total_stock_count_target) > 1:
        raise ValueError('TotalStockCountTarget 仅允许一个')

    if total_stock_count_target:
        t = total_stock_count_target[0]
        if len(selected) > t.count:
            raise ValueError('根据行业选股要求选择股票后，股票数目超过了总股票数目')
        need = t.count - len(selected)
        for pool in pools:
            selected += pool.index.tolist()[:need]
            need -= len(pool.index)
            if need <= 0:
                break

        if t.count > len(selected):
            raise ValueError('候选股票池股票数不够')

    return selected
