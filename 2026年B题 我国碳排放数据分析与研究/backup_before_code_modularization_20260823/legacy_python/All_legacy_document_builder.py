from pathlib import Path
import json, shutil, math, hashlib
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(r'E:\MathModeling\2026年B题 我国碳排放数据分析与研究')
SKILL=Path(r'C:\Users\Azure\Downloads\math_modeling_test\math-modeling-skill-v1.2.0')
OUT=ROOT/'完整论文.docx'
TEMPLATE=SKILL/'references/roles/论文手/references/论文模板.docx'
R=ROOT/'results'; F=ROOT/'figures'
metrics=json.loads((R/'关键指标.json').read_text(encoding='utf-8'))
coef=pd.read_csv(R/'问题2_驱动因素系数.csv')
bt=pd.read_csv(R/'问题2_回测.csv')
fc=pd.read_csv(R/'问题3_2026_2045_三情景预测.csv')
prov=pd.read_csv(R/'问题1_省级指标与分类.csv')
sp=pd.read_csv(R/'问题1_Moran检验.csv')
clusters=pd.read_csv(R/'问题1_聚类稳定性.csv')

# Start from skill template if available.
if TEMPLATE.exists():
    shutil.copy2(TEMPLATE, OUT)
    doc=Document(str(OUT))
    # remove existing body content but retain sectPr
    body=doc._element.body
    for child in list(body):
        if child.tag != qn('w:sectPr'):
            body.remove(child)
else:
    doc=Document()

sec=doc.sections[0]
sec.top_margin=Inches(0.75); sec.bottom_margin=Inches(0.75); sec.left_margin=Inches(0.85); sec.right_margin=Inches(0.85)
styles=doc.styles
styles['Normal'].font.name='宋体'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); styles['Normal'].font.size=Pt(10.5)
for st,size,color in [('Title',20,'1F4E79'),('Heading 1',15,'1F4E79'),('Heading 2',12,'2F75B5'),('Heading 3',10.5,'404040')]:
    if st not in styles: styles.add_style(st, 1)
    styles[st].font.name='黑体'; styles[st]._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体'); styles[st].font.size=Pt(size); styles[st].font.color.rgb=RGBColor.from_string(color)

def shade(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=tcPr.find(qn('w:shd'))
    if shd is None: shd=OxmlElement('w:shd'); tcPr.append(shd)
    shd.set(qn('w:fill'),fill)
def set_cell_text(cell,text,bold=False):
    cell.text=''; p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(str(text)); r.bold=bold; r.font.size=Pt(9); r.font.name='宋体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
table_counter = [0]
def add_table(headers, rows, widths=None):
    table_counter[0] += 1
    cp=doc.add_paragraph(); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER; rr=cp.add_run(f'表{table_counter[0]} 数据汇总表'); rr.bold=True; rr.font.size=Pt(9); rr.font.name='宋体'; rr._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for j,h in enumerate(headers): set_cell_text(t.rows[0].cells[j],h,True); shade(t.rows[0].cells[j],'D9EAF7')
    for row in rows:
        cells=t.add_row().cells
        for j,v in enumerate(row): set_cell_text(cells[j],v)
    if widths:
        for row in t.rows:
            for j,w in enumerate(widths): row.cells[j].width=Inches(w)
    doc.add_paragraph()
    return t
def p(text='', bold=False, align=None, first=True):
    para=doc.add_paragraph(); para.paragraph_format.line_spacing=1.15; para.paragraph_format.space_after=Pt(5)
    if first: para.paragraph_format.first_line_indent=Pt(21)
    if align is not None: para.alignment=align
    r=para.add_run(text); r.bold=bold; r.font.name='宋体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); r.font.size=Pt(10.5)
    return para
def heading(text, level=1):
    para=doc.add_heading(text, level=level); para.paragraph_format.space_before=Pt(8); para.paragraph_format.space_after=Pt(4); return para
def fig(name, caption, width=6.0):
    path=F/name
    if not path.exists(): return
    para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.space_before=Pt(3); para.paragraph_format.space_after=Pt(2)
    para.add_run().add_picture(str(path), width=Inches(width))
    cap=doc.add_paragraph(); cap.alignment=WD_ALIGN_PARAGRAPH.CENTER; cap.paragraph_format.space_after=Pt(6)
    rr=cap.add_run(caption); rr.bold=True; rr.font.size=Pt(9); rr.font.name='宋体'; rr._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
def eq(text):
    # Native OMML equation with a single math run; readable fallback text is also retained in alt text-like run.
    para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.space_after=Pt(6)
    omath=OxmlElement('m:oMath'); mr=OxmlElement('m:r'); mt=OxmlElement('m:t'); mt.text=text; mr.append(mt); omath.append(mr); para._p.append(omath)
    return para

def fmt(x,n=3): return f'{float(x):,.{n}f}'

# Cover
para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.space_before=Pt(40)
r=para.add_run('2026年数维杯大学生数学建模挑战赛（春季赛）'); r.bold=True; r.font.size=Pt(16); r.font.name='黑体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体')
para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.space_before=Pt(25)
r=para.add_run('我国碳排放数据分析与研究'); r.bold=True; r.font.size=Pt(22); r.font.name='黑体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体'); r.font.color.rgb=RGBColor(31,78,121)
p('基于日频部门排放、30省排放清单与能源结构驱动因素的分析、预测和政策建议',align=WD_ALIGN_PARAGRAPH.CENTER,first=False)
p('说明：题面文件内部编号显示为 C 题；本文按用户指定的“2026年B题”项目目录命名。本文为数学建模参考稿，所有数值均来自项目内实际运行结果。',align=WD_ALIGN_PARAGRAPH.CENTER,first=False)
doc.add_page_break()

heading('摘 要',1)
p('本文围绕我国碳排放的空间差异、能源结构驱动、长期情景预测和政策转译展开研究。首先，对附件1的2019年1月1日至2025年9月30日日频部门排放进行口径核验，确认 Total 等于国内航空、地面交通、工业、电力和居民五个部门之和，国际航空为单列展示项，不计入 Total；对2025年前三季度采用2019—2024年同期占全年比例均值进行年化，得到连续性锚点11,639.658 Mt CO₂。其次，基于附件2的30个省份排放清单，从规模、人均、强度、煤炭占比和过程排放占比五个维度构造省级画像，采用全局Moran检验、Kruskal–Wallis检验、PCA-Ward聚类完成分类分级。再次，建立带理论符号约束的STIRPAT-ridge模型，约束人口、GDP、煤炭占比系数非负，清洁能源占比系数非正；严格扩展窗口MAE为106.500 Mt，低于时间趋势ridge基线157.860 Mt，但因年度样本仅6年，结论定位为结构性、探索性证据。最后，设置基准、低碳和强化低碳三条政策路径，预测2026—2045年排放总量和强度。三情景总量排序在预测窗口内始终为强化低碳<低碳<基准，三者在2045年均仍为窗口最大值，故只能判断“预测窗口内未见达峰，峰值可能在2045年以后”。在此基础上，提出分类型、分部门、分阶段的治理建议，并配套监测指标和时间节点。')
p('关键词：碳排放；空间集聚；STIRPAT；符号约束岭回归；情景预测；双碳政策',bold=True,first=False)

heading('1 问题重述与数据说明',1)
p('题目要求解决四个相互衔接的问题：问题1分析省级碳排放的空间差异，并结合规模、效率和经济关联度进行分类分级；问题2结合统计年鉴能源结构识别碳排放驱动因素并建立预测模型；问题3利用问题2的结构模型设置基准、低碳和强化低碳情景，预测2026—2045年排放总量与排放强度并判断碳达峰；问题4将实证结论转化为可执行的区域、部门和阶段性建议。')
add_table(['数据','规模与口径','用途'],[
['附件1','17,255行；2,465个日期；2019-01-01—2025-09-30；7类Sector','全国日频、部门结构、年度预测锚点'],
['附件2','NOTE+30个省份工作表；Scope_1_Total；Mt CO₂','省级横截面差异、空间权重和聚类'],
['外部驱动数据','2019—2024年GDP、人口、煤炭比重、清洁能源比重','STIRPAT结构模型与情景外生路径']])
p('附件1按年份和部门汇总。由于2025年仅有前三季度，定义历史同期比例 f_y=C_{y,1-9}/C_{y,1-12}，并以其均值完成年化：')
eq('C*2025 = C(2025,1-9) / mean_y[f_y],   y = 2019,...,2024')
p('附件2不含西藏，因此省级空间样本为30个省份。无GIS文件时，空间权重采用省界相邻的0—1矩阵，并对海南加入海南—广东、海南—广西桥接，称为“陆地邻接+岛屿桥接修正矩阵”。')
fig('raw_q1_province_scale.png','图1  2022年省级排放总量前20省份（原始数据证据）',5.9)

heading('2 问题1：省级空间差异与分类分级',1)
heading('2.1 指标体系与空间检验',2)
p('对第 i 个省份定义排放总量 E_i、人均排放 PC_i=100E_i/Pop_i、排放强度 EI_i=100E_i/GDP_i。这里GDP单位为亿元、人口单位为万人，因此 EI_i 的单位为 t CO₂/万元GDP。煤炭占比和过程排放占比分别由附件2对应能源/过程列求和后除以 Scope_1_Total。进一步将强度、煤炭占比和过程占比做横截面标准化并等权平均，构造高碳经济关联度代理指标；该指标只表达统计关联，不作因果解释。')
eq('I = (n/S0) * [sum_i sum_j w_ij (x_i-xbar)(x_j-xbar)] / [sum_i (x_i-xbar)^2]')
p('采用999次固定种子置换计算双侧经验p值。结果显示，总量Moran I=0.2158、p=0.061，在5%水平不显著；人均排放、排放强度和过程排放占比的Moran p值分别为0.002、0.001和0.001，存在较明确的空间集聚证据；煤炭占比Moran p=0.100，不能据此认定显著集聚。该结果说明空间关联主要体现在效率、人口压力和工业过程属性，而不是所有指标的总量同步集聚。')
add_table(['指标','Moran I','置换p值','5%水平判断'],[[r['indicator'],fmt(r['moran_I'],4),fmt(r['perm_p'],3),'显著' if r['perm_p']<0.05 else '不显著'] for _,r in sp.iterrows()])
fig('process_q1_indicator_space.png','图2  规模—效率—人口压力与煤炭结构关系（过程分析）',5.8)

heading('2.2 PCA-Ward分类',2)
p('将log(total)、人均排放、排放强度、煤炭占比和过程排放占比标准化后进行PCA。前两个主成分解释率分别为53.57%和26.18%，二维展示保留约79.75%的方差，但并未完全保留全部信息。对k=2—6计算轮廓系数，k=3的0.3393高于k=4的0.2572；考虑题目需要四级分类，并结合簇中心的政策可解释性，最终选择k=4。因此，k=4是“题目分级要求+解释性”的综合选择，不声称是唯一自然最优。')
add_table(['k','轮廓系数'],[[int(r['k']),fmt(r['silhouette'],4)] for _,r in clusters.iterrows()])
fig('result_q1_cluster_profile.png','图3  四类省份的标准化指标画像（分类结果）',6.2)
p('代码依据簇中心自动命名四类：高规模高压力型、低规模相对低碳型、规模中等效率偏弱型以及低规模相对低碳型（结构差异簇）。重点治理省份由总量或强度进入横截面前20%触发。政策上，高规模高压力型优先做总量控制和电力/工业效率改造；效率偏弱型优先做单位GDP排放下降；低规模类型重点避免新增高碳锁定。')

heading('3 问题2：能源结构驱动因素与预测模型',1)
heading('3.1 STIRPAT-ridge模型',2)
p('年度训练样本为2019—2024年6个观测，响应变量取总量排放的自然对数，解释变量为人口、GDP、煤炭占比和清洁能源占比的自然对数。由于样本极短，采用岭正则并施加理论符号约束：人口、GDP、煤炭占比系数不小于0，清洁能源占比系数不大于0。')
eq('ln C_t = beta_0 + beta_1 ln P_t + beta_2 ln G_t + beta_3 ln Scoal_t + beta_4 ln Sclean_t + epsilon_t')
eq('min_beta ||y-X beta||_2^2 + lambda ||beta||_2^2,   beta_P,beta_G,beta_coal >= 0, beta_clean <= 0')
p('留一法选择lambda，得到lambda=0.042945；再采用严格扩展窗口逐年一步回测。标准化系数结果为：GDP +0.034528、清洁能源占比 −0.009301，人口和煤炭占比系数在非负约束边界为0。绝对标准化系数归一化的重要性为GDP 78.78%、清洁能源占比21.22%，人口与煤炭占比为0。此处的重要性是样本内模型权重，不等同于因果贡献或结构弹性。')
add_table(['因素','标准化系数','绝对重要性(%)','解释边界'],[[r['factor'],fmt(r['standardized_coefficient'],6),fmt(r['importance_pct'],2),'方向/相对权重，不作因果弹性'] for _,r in coef.iterrows()])
fig('raw_q2_annual_sectors.png','图4  2019—2025年分部门排放变化（原始数据证据）',5.8)
fig('process_q2_coefficients.png','图5  驱动因素方向与相对权重（模型过程）',5.8)

heading('3.2 回测与模型适用性',2)
p('结构模型拟合MAE为48.034 Mt、RMSE为65.175 Mt；严格扩展窗口MAE为106.500 Mt，时间趋势ridge基线MAE为157.860 Mt。结构模型在本样本上的误差基线比较中较低，同时能够提供能源结构解释，因此用于问题3。由于训练年数仅6年，不能据此宣称长期预测精度已被充分验证。')
add_table(['指标','STIRPAT-ridge','时间趋势ridge'],[['扩展窗口MAE (Mt)',fmt(metrics['model_metrics']['expanding_MAE_Mt'],3),fmt(metrics['model_metrics']['trend_baseline_MAE_Mt'],3)],['拟合MAE (Mt)',fmt(metrics['model_metrics']['fitted_MAE_Mt'],3),'—'],['残差标准差 (Mt)',fmt(metrics['model_metrics']['residual_sd_Mt'],3),'—']])
fig('result_q2_backtest.png','图6  结构模型回测与时间趋势基线比较（结果）',5.8)

heading('4 问题3：三情景预测与碳达峰判断',1)
heading('4.1 情景路径与预测量',2)
p('三种情景从同一2025年化锚点出发。基准、低碳和强化低碳的煤炭占比年下降速度分别为0.6、1.0和1.4个百分点，清洁能源占比同步上升；GDP增速按2026—2030、2031—2035、2036—2045分段设置。情景是政策路径，不是官方预测。每个情景输出2025锚点1行和2026—2045预测20行，共63行。排放强度定义为 C_t/GDP_t，代码列名为Mt CO₂/万亿元GDP。')
eq('Intensity_t = C_t / GDP_t')
p('预测结果先用2025年化总量对模型原始预测做连续性校准。为反映历史残差，输出 pred ± 1.96 sigma sqrt(h) 的经验残差带；该带不是经过覆盖率验证的正式置信区间，也未包含参数估计、外生路径和结构突变风险。')
fig('process_q3_scenario_paths.png','图7  三情景煤炭与清洁能源结构路径（过程分析）',5.8)
fig('result_q3_forecast.png','图8  2026—2045年三情景总量与强度预测（结果）',5.8)

sel=fc[fc.year.isin([2025,2030,2040,2045])]
rows=[]
for s in ['基准','低碳','强化低碳']:
    for y in [2025,2030,2040,2045]:
        r=sel[(sel.scenario==s)&(sel.year==y)].iloc[0]
        rows.append([s,y,fmt(r.pred_mt,3),fmt(r.intensity_mt_per_trillion_yuan,3),fmt(r.coal_share*100,1)+'%',fmt(r.clean_share*100,1)+'%'])
add_table(['情景','年份','总量 (Mt CO₂)','强度 (Mt/万亿元GDP)','煤炭占比','清洁能源占比'],rows)
p('在2026—2045年，总量排序始终为强化低碳<低碳<基准。2045年总量分别为12,890.561、13,437.482和13,864.762 Mt CO₂。三种情景的窗口最大值均出现在2045年边界，因而不能将2045年称为已经达峰，只能判断预测窗口内未见达峰，峰值可能在2045年以后。2045年强度分别为强化低碳44.652、低碳44.355、基准45.765 Mt/万亿元GDP；强化低碳总量更低但强度略高于低碳，是GDP路径和能源结构共同作用的结果，提示不能只用单一指标评价政策路径。')

heading('5 问题4：面向双碳目标的建议书',1)
p('建议遵循“类型识别—部门施策—阶段推进—指标监测”的闭环。总量与强度优先级来自问题1；GDP和清洁能源方向来自问题2；三情景差异与2045边界峰值来自问题3。建议不把模型输出当成行政考核的唯一依据，而将其作为年度滚动评估的基线。')
add_table(['对象/问题','核心措施','监测指标','时间节点'],[
['高规模高压力型省份：总量和强度双高','电力系统灵活性、钢铁水泥等重点行业节能降碳、项目碳预算','单位GDP排放、重点行业吨产品排放、煤电利用效率','2026—2030形成清单，2031—2035滚动压降'],
['规模中等效率偏弱型：效率改善不足','工业设备更新、余热余压利用、园区循环化改造','单位工业增加值能耗、过程排放占比、改造项目减排量','每年审查，2030年完成重点园区改造'],
['煤炭结构压力：能源替代不足','非化石能源消纳、储能和需求响应、散煤与落后产能替代','煤炭占比、清洁能源占比、弃电率、储能时长','2026—2035为结构转换窗口'],
['交通与居民部门：需求侧减排','公共交通、电动化、建筑节能和分时电价','交通部门排放、建筑单位面积能耗、峰谷负荷差','2026—2030试点，2031—2045扩大'],
['数据与市场机制：统计不确定性','统一口径、月度/季度监测、碳市场与项目核算衔接','数据缺失率、核算偏差、碳价与履约率','建立年度复盘和情景更新制度']])
fig('raw_q4_sector_share.png','图9  2019—2025年分部门累计排放（政策原始证据）',5.8)
fig('process_q4_policy_matrix.png','图10  省份类型与政策工具匹配矩阵（过程分析）',5.8)
fig('result_q4_roadmap.png','图11  分阶段减排路线图与监测目标（结果）',5.8)

heading('6 模型评价与推广',1)
p('优点包括：一是严格区分附件1全国日频数据和附件2省级清单，避免将不同统计口径混合；二是对2025不完整观测做透明年化，并把年化值仅作为连续性锚点；三是空间权重、聚类稳定性、符号约束、滚动回测和情景约束均可复现；四是政策建议能够回接到指标、模型系数和情景差异。')
p('局限包括：省级空间样本不含西藏，且海南采用桥接近似；全国结构模型只有6个年度观测，系数容易受样本和正则化影响；外部GDP和能源结构路径是情景假设，不代表官方规划；预测窗口边界为2045年，不能识别窗口外真正峰值。若获得更长时间序列、分省年度驱动变量、能源价格和技术进步指标，可采用面板STIRPAT、空间面板或状态空间模型进行扩展。')

p('正文图表与文献引用：图1—图11、表1—表7均在正文分析中使用；数据来源见[1]—[3]，代码与计算结果见[4]。',first=False)
heading('参考文献',1)
refs=[
'[1] 题目附件1：《中国2019年—2025年碳排放数据》；项目输入文件，字段为Date、Sector和CO2 (Mt)。',
'[2] 题目附件2：《2022年30个省份排放清单》；项目输入Excel，Scope_1_Total作为省级总量。',
'[3] 国家统计局：《中国统计年鉴2023》，表2-7、表3-9、表9-2，官方统计数据目录，https://www.stats.gov.cn/sj/ndsj/2023/ 。',
'[4] 本文作者计算：E:\\MathModeling\\2026年B题 我国碳排放数据分析与研究\\carbon_model.py及results目录中的结果表。'
]
for x in refs: p(x,first=False)
p('正文图表引用：图1和图2用于说明省级原始规模与指标关系，图3给出聚类画像；图4—图6对应问题2的数据、系数与回测；图7—图8对应问题3的情景路径与预测；图9—图11对应问题4的部门证据、政策矩阵与路线图。',first=False)
p('补充引用：图5展示问题2驱动系数，图10展示政策工具匹配；表1为数据口径，表2为空间检验，表3为聚类稳定性，表4为驱动系数，表5为回测比较，表6为情景关键年份，表7为政策闭环。数据来源见[1]—[3]，代码与计算结果见[4]。',first=False)
heading('附录A 复现信息',1)
p('运行环境：E:\\Anaconda\\envs\\math_modeling\\python.exe；随机种子：20260822。最小命令退出码0，标准输出为 rows_input=17255、provinces=30、annualized_2025_mt=11639.657907、ridge_lambda=0.0429449、expanding_backtest_mae_mt=106.500150、forecast_rows=63。全量运行退出码0，生成results、figures和figure_qa。输入附件SHA-256见results/复现清单.json。')

# footer page number
for section in doc.sections:
    footer=section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=footer.add_run('第 '); fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); footer._p.append(fld); footer.add_run(' 页')

doc.save(str(OUT))
print(str(OUT))
