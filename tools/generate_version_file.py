# tools/generate_version_file.py
# Generate PyInstaller version file (Windows resource) -> version_file.txt
# Ambil metadata dari satu sumber supaya konsisten.

import re
from pathlib import Path

# === Metadata aplikasi ===
APP_NAME      = "LeafLabeler"
APP_VERSION   = "1.0.0"  # format: major.minor.patch
APP_PUBLISHER = "OS AI Corp"
COPYRIGHT_TXT = "© 2025 Oki Saputra - OS AI Corp. All rights reserved."
ORIGINAL_EXE  = "LeafLabeler.exe"

def parse_semver(v: str):
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", v)
    if not m:
        return (1,0,0)
    return tuple(int(x) for x in m.groups())

def main():
    ver_tuple = parse_semver(APP_VERSION)
    filevers  = f"({ver_tuple[0]}, {ver_tuple[1]}, {ver_tuple[2]}, 0)"
    prodvers  = filevers
    ver_str   = f"{ver_tuple[0]}.{ver_tuple[1]}.{ver_tuple[2]}.0"

    txt = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={filevers},
    prodvers={prodvers},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '040904B0',
          [
            StringStruct('CompanyName', '{APP_PUBLISHER}'),
            StringStruct('FileDescription', '{APP_NAME}'),
            StringStruct('FileVersion', '{ver_str}'),
            StringStruct('InternalName', '{APP_NAME}'),
            StringStruct('LegalCopyright', '{COPYRIGHT_TXT}'),
            StringStruct('OriginalFilename', '{ORIGINAL_EXE}'),
            StringStruct('ProductName', '{APP_NAME}'),
            StringStruct('ProductVersion', '{ver_str}')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    Path("version_file.txt").write_text(txt, encoding="utf-8")
    print("OK -> version_file.txt")

if __name__ == "__main__":
    main()
