"""Strict, version-aware interpretation of BIDS Validator JSON reports."""


def parse_bids_report(report):
    if not isinstance(report, dict) or not isinstance(report.get("issues"), dict):
        raise ValueError("Missing or malformed BIDS issues object")
    issues = report["issues"]
    if "issues" in issues:
        entries = issues["issues"]
        if not isinstance(entries, list):
            raise ValueError("BIDS v3 issues must be a list")
        grouped = {name: [] for name in ["error", "warning", "info", "ignore"]}
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("severity") not in grouped:
                raise ValueError("Unknown BIDS issue severity or malformed issue")
            grouped[entry["severity"]].append(entry)
        errors, warnings = grouped["error"], grouped["warning"]
        format_name = "schema-validator-v3"
    elif "errors" in issues and "warnings" in issues:
        errors, warnings = issues["errors"], issues["warnings"]
        if not isinstance(errors, list) or not isinstance(warnings, list):
            raise ValueError("Legacy BIDS errors and warnings must be lists")
        if any(not isinstance(entry, dict) for entry in errors + warnings):
            raise ValueError("Malformed legacy BIDS issue")
        format_name = "legacy-validator-v1"
    else:
        raise ValueError("Unknown BIDS report layout; refusing to infer a pass")
    return {"format": format_name, "error_count": len(errors), "warning_count": len(warnings),
            "errors": errors, "warnings": warnings, "passed": not errors}
