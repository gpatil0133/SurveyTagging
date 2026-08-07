"""Corporate/tenant-level context model."""

from pydantic import BaseModel, Field


class CorporateContext(BaseModel):
    """Tenant-level metadata extracted from {TenantID}_CorporateData.json."""

    corporate_no: int = 0
    corporate_id: str = ""
    corporate_name: str = ""
    first_name: str = ""
    last_name: str = ""
    job_title: str = ""
    email_address: str = ""

    # Location
    country: str = ""
    country_name: str = ""
    state: str = ""
    city: str = ""

    # Business profile
    industry: str = ""
    department: str = ""
    purpose: str = ""
    corporate_size: str = ""
    revenue: str = ""

    # Feature flags
    is_cx: bool = False
    is_engage: bool = False
    is_wlc: bool = False
    is_enterprise_plus: bool = False
    account_type: int = 0
    sub_account_type: int = 0
    lang: str = "en"

    @property
    def owner_display_name(self) -> str:
        """Full name of the account owner for auto-suggestion."""
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p).strip()

    @property
    def is_empty(self) -> bool:
        """Whether this context was created from a missing corporate file."""
        return self.corporate_no == 0 and self.corporate_name == ""
