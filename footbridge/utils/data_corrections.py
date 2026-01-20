def fix_scunthorpe_wealdestone_2025_2026(df):
    # Fix for specific incorrect data from source
    # Scunthorpe Utd vs Wealdstone on 2025-09-06 should be 2-1, not 0-0
    mask = (
        (df["Date"] == "2025-09-06")
        & (df["Home"] == "Scunthorpe Utd")
        & (df["Away"] == "Wealdstone")
        & (df["League"] == "National-League")
        & (df["Season"] == "2025-2026")
    )
    df.loc[mask, "FTHG"] = 2
    df.loc[mask, "FTAG"] = 1

    return df
