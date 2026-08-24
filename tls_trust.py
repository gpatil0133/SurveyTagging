"""Make Python's TLS verification use the OS trust store.

The SoGo hosts (intuc especially) serve certificates that chain to a corporate
root CA. That root is installed in the Windows certificate store, so a browser
and `Invoke-WebRequest` reach the host without complaint — but Python does not
read the Windows store. httpx verifies against the `certifi` bundle, which
contains only public roots, and every outbound call dies with:

    [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    unable to get local issuer certificate

`truststore` fixes this by patching `ssl.SSLContext` to verify through the
platform's own store — the same trick pip uses. It has been a declared
dependency and installed all along, and both settings.py and .env.example
promised it worked "automatically", but nothing in the app ever called
`inject_into_ssl()`. Only the standalone probe in tests/test_smx.py did, which
is exactly why the probe could reach apismx while the service could not.

This module is that missing call. It is deliberately not a settings validator
or an import-time side effect: patching the ssl module is process-global and
affects every TLS connection in it (SoGo, Anthropic, anything), so it happens
once, in the open, from the composition root.

Not called when TLS verification is off, or when an explicit CA bundle is
configured — those are the other two branches of the three-way choice
settings.py documents, and each already works on its own terms.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("survey_tagging.tls")

# inject_into_ssl() is global and not meant to be applied twice; every entry
# point calls install() so the second and later calls have to be no-ops.
_installed = False


def install(settings: object) -> str:
    """Route TLS verification through the OS trust store. Returns what happened.

    The return value is for logging and for tests — callers do not branch on it.
    Never raises: a service that cannot verify certificates should fail on the
    call that needs one, with that call's own error, not at startup.
    """
    global _installed

    if _installed:
        return "already-installed"

    # Explicit CA bundle wins. It is the production-clean option, it is already
    # honoured by SmxClient's `verify=`, and truststore's SSLContext does not
    # treat load_verify_locations() the way the stdlib one does — so injecting
    # underneath a configured bundle would quietly change which roots are
    # trusted. Leave that path exactly as documented.
    if getattr(settings, "sogo_ca_bundle_path", ""):
        return "skipped-ca-bundle-configured"

    if not getattr(settings, "sogo_verify_ssl", True):
        # Verification is off; there is nothing for a trust store to do.
        return "skipped-verification-disabled"

    try:
        import truststore
    except ImportError:
        # Declared in pyproject, so this means an incomplete install rather than
        # a supported configuration. WARNING, with the fix in the message: the
        # symptom otherwise arrives much later as an opaque CERTIFICATE_VERIFY_
        # FAILED from whichever call happens to run first.
        logger.warning(
            "tls_truststore_missing",
            extra={"remedy": "pip install truststore, or set "
                             "SURVEY_TAGGER_SOGO_CA_BUNDLE_PATH to the corporate "
                             "root CA as PEM"},
        )
        return "unavailable"

    try:
        truststore.inject_into_ssl()
    except Exception as e:  # noqa: BLE001 - startup must survive anything here
        logger.warning("tls_truststore_inject_failed", extra={"error": str(e)})
        return "failed"

    _installed = True
    logger.info("tls_truststore_installed")
    return "installed"


def _reset_for_tests() -> None:
    """Clear the once-only latch. Tests only — nothing in the app calls this."""
    global _installed
    _installed = False
