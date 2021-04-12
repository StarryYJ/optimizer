from rqoptimizer import *
import rqfactor
import rqdatac
import rqrisk
import rqalpha
from rqdatac import *

rqdatac.init('license',
			 'NSmWf24LQ53L8v1TKo-xq3_glq-Mq1RIJg81t6oKZnCW07ZrZvrBEBArPoo48ozXupXChlWEd6lB3C3nEpm83mQBvj_EVg92dxEJSR4XOD8EK76_aknPz1ZO1xHMnL_eTP11I8PSGEKbqcH-TYhLOC4_MC0-6cWgSxlAFe9q39M=XEb-4930b4g05BOoCGdX0zMDV4yxKQrPjWfAjkPv-PL5AHqAVQ8NYkYV9ma5HWxyVp6aXE7s5S7bYhkFSTGM21aV6y68L39IfH9vhHr0lDJje0cVNQcn5V1TxzPRnxqZuNaM20loa02Ij-30Qbv4X08JBUaYtE3MQKrSDTOzcI8=',
			 ("rqdatad-pro.ricequant.com", 16011))

order_book_ids = ['600969', '300649', '603037', '002111']
date = '2021-04-05'

portfolio_optimize(order_book_ids, date, objective=MinVariance(),
				   bnds=None, cons=None, benchmark=None, cov_model=CovModel.FACTOR_MODEL_DAILY)
rqdatac.index_weights()