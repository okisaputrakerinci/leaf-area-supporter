import re, sys, io, os

p = r".\gui_labeler.py"
with open(p, "r", encoding="utf-8") as f:
    s = f.read()

# Perbaiki kasus umum string yang terputus pada f.write(...)
s2 = s
s2 = s2.replace('f.write(line + "', 'f.write(line + "\\n")')
s2 = s2.replace('f.write(c + "',    'f.write(c + "\\n")')

# (opsional) rapikan newline ke LF
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(s2)

# Validasi sintaks
import py_compile
py_compile.compile(p, doraise=True)
print("Patched & syntax OK")
