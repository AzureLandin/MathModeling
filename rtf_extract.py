import re, sys
from pathlib import Path
p=Path(sys.argv[1])
s=p.read_bytes().decode('latin1')
# remove header/font/style destinations and control groups only roughly
# decode hex escapes as codepage 936 bytes
s=re.sub(r"(?:\\'[0-9a-fA-F]{2})+", lambda m: bytes(int(x,16) for x in re.findall(r"\\'([0-9a-fA-F]{2})",m.group(0))).decode('gbk','replace'), s)
# decode unicode escapes
s=re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) if int(m.group(1))>=0 else int(m.group(1))+65536), s)
# RTF controls
s=re.sub(r'\\(par|line|tab)\b', lambda m: '\n' if m.group(1)!='tab' else '\t', s)
s=re.sub(r'\\[a-zA-Z]+-?\d* ?', '', s)
s=s.replace('\\{','{').replace('\\}','}').replace('\\\\','\\')
# remove groups that are destinations / metadata, repeat balanced-ish
s=re.sub(r'\{\\\*[^{}]*\}', '', s)
s=re.sub(r'[{}]', '', s)
# normalize
s='\n'.join(line.strip() for line in s.splitlines() if line.strip())
print(s)
