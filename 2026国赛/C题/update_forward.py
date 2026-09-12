from pathlib import Path
root=Path.cwd()
entry='**2026-09-13问题三日内负载修正前向实验接收核查：** 已完成保存结果有限核查，意见见 `reports/问题三/问题三_日内负载修正前向预测对照实验接收核查.md`，脚本 `code/q3_intraday_load_forward_review.py`，证据 `results/q3_intraday_load_forward_review/review.json`。11项登记哈希全匹配；73146条B集预测与冻结档案一致，A集36072个唯一目标，逐段F0/F1/F2复算差≤1.2e-13 kW，18个参数组独立闭式复算差≤2.3e-13，历史资格均为过去28日不含当天。A集F0/F1/F2 MAE=142.256752/141.724919/133.114875 kW，F2较F0/F1分别−9.141878/−8.610044 kW，三节点同向改善；但F2仅53.54%目标更准、188天优/146天差，尾槽MAE不优于F0（163.479318 vs 163.041353），正式β约20.30%为负且范围−1.179065—3.660811。结论：结果支持进入一次受控F0/F2费用对照，不支持直接采纳；必须为F2自身重建发布误差q75、各自连续SOC运行，不能复用L75保护或轨迹。本次未训练/求解/回放/q75/填表，图件两张已目视通过。'
for name in ['建模上下文记忆.md','reports/问题三/问题三_当前建模思路.md','reports/项目进度.md','reports/00_报告导航.md']:
    p=root/name;s=p.read_text(encoding='utf-8-sig'); i=s.find('\n')+1;p.write_text(s[:i]+'\n'+entry+'\n'+s[i:],encoding='utf-8')
