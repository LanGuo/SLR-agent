# slr_agent/prisma.py


def generate_prisma_mermaid(
    n_retrieved: int,
    n_duplicates: int,
    n_screened: int,
    n_excluded_abstract: int,
    n_fulltext: int | None,
    n_excluded_fulltext: int | None,
    n_included: int,
    n_quarantined: int,
) -> str:
    """Generate a PRISMA 2020 flow diagram as a Mermaid flowchart string."""
    lines = ["```mermaid", "flowchart TD"]
    lines.append(f'    ID["Identification<br/>Records from PubMed<br/>n = {n_retrieved}"]')
    lines.append(f'    DUP["After duplicate removal<br/>n = {n_retrieved - n_duplicates}"]')
    lines.append(f'    SCR["Records screened<br/>n = {n_screened}"]')
    lines.append(f'    EXA["Excluded on abstract<br/>n = {n_excluded_abstract}"]')

    if n_fulltext is not None:
        lines.append(f'    FTR["Full-text assessed<br/>n = {n_fulltext}"]')
        lines.append(f'    EXF["Excluded full-text<br/>n = {n_excluded_fulltext or 0}"]')
        lines.append(f'    INC["Included in review<br/>n = {n_included}"]')
        if n_quarantined:
            lines.append(f'    QUA["Quarantined fields<br/>n = {n_quarantined}"]')
        lines += [
            "    ID --> DUP --> SCR",
            "    SCR --> EXA",
            "    SCR --> FTR",
            "    FTR --> EXF",
            "    FTR --> INC",
        ]
        if n_quarantined:
            lines.append("    INC --> QUA")
    else:
        lines.append(f'    INC["Included in review<br/>n = {n_included}"]')
        if n_quarantined:
            lines.append(f'    QUA["Quarantined fields<br/>n = {n_quarantined}"]')
        lines += [
            "    ID --> DUP --> SCR",
            "    SCR --> EXA",
            "    SCR --> INC",
        ]
        if n_quarantined:
            lines.append("    INC --> QUA")

    lines.append("```")
    return "\n".join(lines)
