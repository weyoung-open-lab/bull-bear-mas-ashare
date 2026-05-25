"""Bull-Bear 对抗框架 v1。

Alpha Agent (Trend) 给出 bull_score（看涨证据）
Bear Agent 学习 max_drawdown_5d_z 给出 bear_score（看跌风险）
conviction(i, t) = bull(i, t) − α × bear(i, t)
"""
