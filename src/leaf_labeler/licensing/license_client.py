# src/leaf_labeler/licensing/license_client.py

from leaf_labeler.meta import APP_NAME, APP_PUBLISHER  # APP_PUBLISHER opsional
import os, json, base64, datetime, hashlib, hmac, uuid, ctypes
from typing import Optional, Tuple, List

PRODUCT_NAME   = "LeafLabeler"
PUBLISHER_NAME = "Your Org"
TRIAL_DAYS     = 14
SCHEMA_VER     = 1

APPDATA_LOCAL = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
PROGRAMDATA   = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
ORG_KEY       = "YourOrg"  # tanpa spasi

ANCHORS = {
    "HKCU": rf"HKCU\Software\{ORG_KEY}\{PRODUCT_NAME}",
    "HKLM": rf"HKLM\Software\{ORG_KEY}\{PRODUCT_NAME}",
    "PD"  : os.path.join(PROGRAMDATA, ORG_KEY, f".{PRODUCT_NAME.lower()}_lic"),
    "LA"  : os.path.join(APPDATA_LOCAL, ORG_KEY, f".{PRODUCT_NAME.lower()}_lic"),
}

# GANTI ini (32 byte random) sebelum rilis
HMAC_SECRET = b"REPLACE_WITH_RANDOM_32_BYTE_SECRET____________"

# GANTI public key RSA 2048-mu sebelum rilis
PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0XtKGNDPZi4I1QDmln0Y
Gn4Uwsk122KxXJv5HxOf6KG6iqZWJbs7kUeS9efWLqEVudosuAsBCsqLo2Npxg10
Cw9zt/ScSQWk+TXEGlpTC8A+LMZVEgLXCxqN6cRm5+2VBitPAC9gYz0FUJTexdzD
od9fyJMM6lbwVbbhDtTeXaIuOTKMceb058g8F7sbEE9qx2trDdoAWDKt0pr1Ln/p
nY+fVjm4xqf5SR8xJUmENZB6BXlvOj6GJ41QdExMHtYeq0sLuTq/WHWjmAfhrqwi
QthL58+O0/n0d62GilMKva5dMGrEhBvsJ17JqBBf3AgVJ9QRB94SsUSODZf8k01f
iQIDAQAB
-----END PUBLIC KEY-----
"""

import sys
if sys.platform != "win32":
    raise RuntimeError("Lisensi ini khusus Windows.")

try:
    import winreg
except Exception:
    pass

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
except Exception as e:
    raise RuntimeError("butuh 'cryptography' (pip install cryptography)") from e

def _sha256s(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def _hmac(d: bytes) -> str:    return hmac.new(HMAC_SECRET, d, hashlib.sha256).hexdigest()
def _today() -> datetime.date: return datetime.date.today()
def _parse_date(s: str) -> datetime.date: return datetime.date.fromisoformat(s)
def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d): os.makedirs(d, exist_ok=True)

# ---- fingerprint ----
def _get_machine_guid() -> str:
    try:
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
        v, _ = winreg.QueryValueEx(k, "MachineGuid"); winreg.CloseKey(k)
        return str(v)
    except Exception:
        return "NO-MGUID"

def _get_volume_serial(drive: str = "C:\\") -> str:
    vol_serial = ctypes.c_uint32()
    max_comp_len = ctypes.c_uint32()
    fs_flags = ctypes.c_uint32()
    vol_name_buf = ctypes.create_unicode_buffer(1024)
    fs_name_buf  = ctypes.create_unicode_buffer(1024)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive), vol_name_buf, 1024,
        ctypes.byref(vol_serial), ctypes.byref(max_comp_len),
        ctypes.byref(fs_flags), fs_name_buf, 1024
    )
    return f"{vol_serial.value:08X}" if ok else "NO-VSER"

def _get_mac() -> str:
    try:
        return f"{uuid.getnode():012x}"
    except Exception:
        return "NO-MAC"

def hardware_fingerprint() -> str:
    raw = (_get_machine_guid() + "|" + _get_volume_serial("C:\\") + "|" + _get_mac()).encode("utf-8")
    return _sha256s(raw)

# ---- registry & file ----
def _reg_set(root, path, name, value) -> bool:
    try:
        hroot = winreg.HKEY_CURRENT_USER if root == "HKCU" else winreg.HKEY_LOCAL_MACHINE
        key = winreg.CreateKey(hroot, path.replace("HKCU\\","").replace("HKLM\\",""))
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value); winreg.CloseKey(key); return True
    except Exception:
        return False

def _reg_get(root, path, name) -> Optional[str]:
    try:
        hroot = winreg.HKEY_CURRENT_USER if root == "HKCU" else winreg.HKEY_LOCAL_MACHINE
        key = winreg.OpenKey(hroot, path.replace("HKCU\\","").replace("HKLM\\",""))
        v, _ = winreg.QueryValueEx(key, name); winreg.CloseKey(key); return str(v)
    except Exception:
        return None

def _file_write(p: str, data: dict) -> bool:
    try:
        _ensure_dir(p)
        with open(p, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def _file_read(p: str) -> Optional[dict]:
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return None

# ---- trial state ----
def _make_trial_blob(start_date, first_seen, last_seen) -> dict:
    payload = {
        "schema": SCHEMA_VER, "product": PRODUCT_NAME,
        "start_date": start_date.isoformat(),
        "first_seen": first_seen.isoformat(),
        "last_seen":  last_seen.isoformat(),
    }
    ser = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["hmac"] = _hmac(ser)
    return payload

def _validate_blob(blob: dict) -> bool:
    try:
        hm = blob.get("hmac",""); cp = dict(blob); cp.pop("hmac", None)
        ser = json.dumps(cp, sort_keys=True).encode("utf-8")
        return hmac.compare_digest(hm, _hmac(ser))
    except Exception:
        return False

def _read_all_anchors() -> List[Tuple[str, dict]]:
    out = []
    s = _reg_get("HKCU", ANCHORS["HKCU"], "Trial")
    if s:
        try: out.append(("HKCU", json.loads(s)))
        except: pass
    s = _reg_get("HKLM", ANCHORS["HKLM"], "Trial")
    if s:
        try: out.append(("HKLM", json.loads(s)))
        except: pass
    for k in ("PD","LA"):
        d = _file_read(ANCHORS[k])
        if d: out.append((k, d))
    return out

def _write_all_anchors(blob: dict):
    _reg_set("HKCU", ANCHORS["HKCU"], "Trial", json.dumps(blob))
    _reg_set("HKLM", ANCHORS["HKLM"], "Trial", json.dumps(blob))
    for k in ("PD","LA"): _file_write(ANCHORS[k], blob)

def _merge_trial_state(blobs: List[Tuple[str, dict]]) -> Optional[dict]:
    valid = [b for _,b in blobs if _validate_blob(b) and b.get("product")==PRODUCT_NAME]
    if not valid: return None
    sd = min(_parse_date(b["start_date"]) for b in valid)
    fs = min(_parse_date(b.get("first_seen", b["start_date"])) for b in valid)
    ls = max(_parse_date(b.get("last_seen",  b["start_date"])) for b in valid)
    return _make_trial_blob(sd, fs, ls)

def _init_or_update_trial(now: datetime.date) -> dict:
    blobs = _read_all_anchors()
    if not blobs:
        blob = _make_trial_blob(now, now, now); _write_all_anchors(blob); return blob
    merged = _merge_trial_state(blobs) or _make_trial_blob(now, now, now)
    fs = _parse_date(merged["first_seen"])
    if (fs - now).days > 2:
        expired = _make_trial_blob(now - datetime.timedelta(days=TRIAL_DAYS+1), fs, now)
        _write_all_anchors(expired); return expired
    ls = _parse_date(merged["last_seen"])
    if now > ls:
        merged = _make_trial_blob(_parse_date(merged["start_date"]), fs, now)
        _write_all_anchors(merged)
    return merged

# ---- license verify (RSA) ----
def _load_public_key():
    return serialization.load_pem_public_key(PUBLIC_KEY_PEM, backend=default_backend())

def _b64url_dec(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s + pad)

def parse_and_verify_license(key_text: str, fp_expected: str) -> Tuple[bool, dict, str]:
    try:
        parts = key_text.strip().split(".")
        if len(parts) != 2: return False, {}, "Format kunci tidak valid"
        payload_b = _b64url_dec(parts[0]); sig_b = _b64url_dec(parts[1])
        payload   = json.loads(payload_b.decode("utf-8"))
        if payload.get("product") != PRODUCT_NAME: return False, payload, "Produk mismatch"
        fp_in = payload.get("fp")
        if fp_in and fp_in != fp_expected: return False, payload, "Fingerprint mismatch"
        exp = payload.get("exp")
        if exp and _today() > _parse_date(exp): return False, payload, "Lisensi kedaluwarsa"
        _load_public_key().verify(sig_b, payload_b, padding.PKCS1v15(), hashes.SHA256())
        return True, payload, ""
    except Exception as e:
        return False, {}, f"Verifikasi gagal: {e}"

# ---- API utama ----
def check_license() -> dict:
    today = _today()
    trial = _init_or_update_trial(today)
    sd = _parse_date(trial["start_date"])
    remaining = max(0, TRIAL_DAYS - (today - sd).days)

    lic_text = None
    for src in ("HKCU","HKLM"):
        s = _reg_get(src, ANCHORS[src], "LicenseKey")
        if s: lic_text = s; break
    if not lic_text:
        for k in ("PD","LA"):
            d = _file_read(ANCHORS[k])
            if d and "LicenseKey" in d: lic_text = d["LicenseKey"]; break

    fp = hardware_fingerprint()
    if lic_text:
        ok, payload, _ = parse_and_verify_license(lic_text, fp)
        if ok:
            return {"valid": True, "type":"full", "remaining": None, "expired": False,
                    "name": payload.get("name"), "features": payload.get("features", [])}

    if remaining <= 0:
        return {"valid": False, "type":"trial", "remaining": 0, "expired": True,
                "name": None, "features": None}
    return {"valid": True, "type":"trial", "remaining": remaining, "expired": False,
            "name": None, "features": None}

def save_license_key(key_text: str) -> Tuple[bool, str]:
    fp = hardware_fingerprint()
    ok, _payload, err = parse_and_verify_license(key_text, fp)
    if not ok: return False, err or "Key invalid"
    _reg_set("HKCU", ANCHORS["HKCU"], "LicenseKey", key_text)
    _reg_set("HKLM", ANCHORS["HKLM"], "LicenseKey", key_text)
    blobs = _read_all_anchors()
    merged = _merge_trial_state(blobs) if blobs else _make_trial_blob(_today(), _today(), _today())
    merged["LicenseKey"] = key_text
    for k in ("PD","LA"): _file_write(ANCHORS[k], merged)
    return True, "Lisensi tersimpan"
