# Derivative-Lab

复现 Hull 的《期权、期货及其衍生品》中的定价模型，用于量化学习与研究。

## Overview

`derivlab` 是一个可安装的 Python 包，实现了书中各章节对应的定价模型实现。具体的实验、图标、讲解、笔记在 `notebooks/` 目录下用 Jupyter Notebook完成。

## Installation

```bash
conda env create -f environment.yml
conda activate derivlab
pip install -e ".[dev,notebook]"
```

## Project Structure

```txt
src/derivlab/
├── pricing/          # 期权定价基础:BSM 公式、CRR 二叉树
└── utils/            # 预留

tests/                # pytest 单元测试,对照已知解析解/例题数值验证
notebooks/            # 交互式实验与讲解
```

## Usage Example

```python
import derivlab as dl

dl.bsm_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
dl.crr_binomial_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, n_steps=200, option_type="put", exercise="american")
```

## Testing

```bash
pytest -v
```

## Roadmap

- [x] 期权定价基础:BSM 公式、CRR 二叉树(European / American)
- [x] 期货与远期定价
- [x] Greeks 与对冲
- [x] 波动率(隐含波动率、波动率微笑)
- [ ] 蒙特卡洛模拟
- [ ] 奇异期权
- [ ] 利率期限结构

## References

- Hull, J. C. _Options, Futures, and Other Derivatives_.

## License

MIT — see [LICENSE](LICENSE).
