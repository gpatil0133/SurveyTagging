#!/usr/bin/env python
"""Diagnose SHARE_ROOT connectivity before starting the service.

    .venv/bin/python check_share.py

Answers, in order: are the settings valid, can we log in, does the configured
path actually exist on the share, what tenants does discovery see, and can we
write. Each step prints what it tried, so a failure names the thing to fix
rather than surfacing later as an empty tenant list — every discovery function
swallows OSError by design and cannot tell "no surveys" from "cannot log in".
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import discovery
import sharefs

OK, BAD, INFO = "  ok ", " FAIL", "     "


def main() -> int:
    # ---- 1. settings ----
    try:
        from settings import Settings

        s = Settings()
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} settings rejected the configuration:\n\n{e}\n")
        return 1

    root = Path(s.share_root) if s.share_root else Path(s.data_dir)
    unc = sharefs.is_unc(root)
    print(f"{INFO} SHARE_ROOT : {root}")
    print(f"{INFO} transport  : {'SMB (no mount)' if unc else 'local filesystem'}")
    if unc:
        print(f"{INFO} server     : {sharefs.server_of(root)}")
        print(f"{INFO} user       : {s.image_user or '(none)'}")
        print(f"{INFO} password   : {'set' if s.image_pass else '(empty)'}")
    print()

    # ---- 2. log in ----
    if unc:
        try:
            sharefs.configure(s.image_user, s.image_pass)
            sharefs.connect(root)
            print(f"{OK} authenticated to {sharefs.server_of(root)}")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} could not open an SMB session: {type(e).__name__}: {e}")
            print(f"{INFO} a LogonFailure here means the username/password is wrong;")
            print(f"{INFO} a domain account is written DOMAIN\\user.")
            return 1

    # ---- 3. walk the path down, so a wrong component is named exactly ----
    parts = root.parts
    start = 3 if unc else 1  # //server/share is the shallowest openable unit
    walked = Path(parts[0]).joinpath(*parts[1:start]) if len(parts) >= start else root
    for part in parts[start:]:
        walked = walked / part
        try:
            if sharefs.is_dir(walked):
                print(f"{OK} {walked}")
            else:
                print(f"{BAD} {walked}  <- not a directory")
                _suggest(walked)
                return 1
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} {walked}  <- {type(e).__name__}: {str(e)[:120]}")
            _suggest(walked)
            return 1
    print()

    # ---- 4. what discovery sees ----
    tenants = discovery.list_tenant_ids(root)
    if not tenants:
        print(f"{BAD} no tenant folders found under {root}")
        print(f"{INFO} a tenant is a NUMERIC folder containing a SurveyData/ subfolder.")
        try:
            names = [p.name for p in sharefs.iterdir(root)][:20]
            print(f"{INFO} what is actually there: {names}")
        except Exception:  # noqa: BLE001
            pass
        return 1
    print(f"{OK} {len(tenants)} tenant(s): {tenants[:10]}{' ...' if len(tenants) > 10 else ''}")
    total = 0
    for tid in tenants[:5]:
        nos = discovery.list_survey_nos(root, tid)
        total += len(nos)
        print(f"{INFO}   tenant {tid}: {len(nos)} survey(s) {nos[:8]}")
    if total == 0:
        print(f"{BAD} tenants exist but none has a survey_structure.json")
        return 1

    # ---- 5. write test, where the pipeline actually writes ----
    tid = tenants[0]
    sno = discovery.list_survey_nos(root, tid)[0]
    probe = discovery.survey_dir(root, tid, sno) / ".sharefs_write_probe.tmp"
    try:
        sharefs.write_text(probe, "probe")
        sharefs.unlink(probe, missing_ok=True)
        print(f"{OK} writable at {probe.parent}")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} NOT writable at {probe.parent}: {type(e).__name__}: {str(e)[:120]}")
        print(f"{INFO} reads would work and tagging would run, but every")
        print(f"{INFO} tagged_output.json write would fail. Grant the account write access.")
        return 1

    print()
    print("     share is reachable, readable and writable — safe to start the service.")
    return 0


def _suggest(failed: Path) -> None:
    """List the parent's children so a typo'd component is obvious."""
    try:
        names = sorted(p.name for p in sharefs.iterdir(failed.parent))
    except Exception:  # noqa: BLE001
        return
    print(f"{INFO} {failed.parent} contains: {names[:30]}")
    print(f"{INFO} fix SURVEY_TAGGER_SHARE_ROOT in .env to match.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
