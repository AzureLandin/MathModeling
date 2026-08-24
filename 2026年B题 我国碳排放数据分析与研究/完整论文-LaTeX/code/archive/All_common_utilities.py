from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SKILL_ROOT = Path(r"C:\Users\Azure\Downloads\math_modeling_test\math-modeling-skill-v1.2.0")
sys.path.insert(0, str(SKILL_ROOT / "tools" / "figure" / "scripts"))
from export_figure import export_figure

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','Arial Unicode MS','DejaVu Sans'],'axes.unicode_minus':False,'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'figure.dpi':120})

COLORS = {'blue':'#2166AC','teal':'#1B9E77','orange':'#D95F02','purple':'#7570B3','red':'#B2182B','gray':'#666666','light':'#F0F0F0'}

def savefig(fig, path, size=(6.5,4.2)):
    export_figure(fig, str(path), formats=['svg','png'], size_inches=size, dpi=300, grayscale_preview=False, tight=True)
    from PIL import Image
    qa=Path(path).parent.parent/'figure_qa'; qa.mkdir(exist_ok=True,parents=True)
    Image.open(str(path)+'.png').convert('L').save(qa/(Path(path).name+'_grayscale.png'),dpi=(300,300))
    plt.close(fig)

def sha256(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
