import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Carrier Removal Tool", layout="wide")

# -----------------------------
# Required columns
# -----------------------------
INPUT_REQUIRED_COLS = ["PracticeId", "ProviderId", "LocationId", "PlanIds", "MappingLevel"]
DB_REQUIRED_COLS = ["Carrier_ID", "Carrier_Name", "Plan_ID", "Plan_Name", "Plan_Type", "Plan Classification"]
DEPRECATED_ACCEPTABLE_PLAN_ID_COLS = ["Plan_ID", "PlanId", "plan_id", "planid"]


# -----------------------------
# Helpers
# -----------------------------
def split_csv_ids(cell_value: str):
    if cell_value is None:
        return []
    s = str(cell_value).strip()
    if not s or s == "-" or s.lower() == "nan":
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def normalize_str(x):
    if x is None:
        return ""
    return str(x).strip()


def read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str).fillna("")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, dtype=str).fillna("")
    raise ValueError("Unsupported file type. Please upload .csv or .xlsx")


def validate_columns(df: pd.DataFrame, required_cols: list[str], label: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def explode_db_plan_ids(db_df: pd.DataFrame) -> pd.DataFrame:
    db_df = db_df.copy()
    for c in DB_REQUIRED_COLS:
        db_df[c] = db_df[c].astype(str).map(normalize_str)

    db_df["_plan_split"] = db_df["Plan_ID"].apply(split_csv_ids)
    db_df = db_df.explode("_plan_split").reset_index(drop=True)
    db_df["Plan_ID"] = db_df["_plan_split"].fillna("").astype(str).map(normalize_str)
    db_df = db_df.drop(columns=["_plan_split"])
    db_df = db_df[db_df["Plan_ID"] != ""].reset_index(drop=True)
    return db_df


def build_plan_lookup(db_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in db_df.iterrows():
        pid = row["Plan_ID"]
        if pid and pid not in lookup:
            lookup[pid] = row.to_dict()
    return lookup


def extract_input_plan_ids(input_df: pd.DataFrame) -> set[str]:
    plan_ids = set()
    for v in input_df["PlanIds"].astype(str).tolist():
        for pid in split_csv_ids(v):
            plan_ids.add(pid)
    return plan_ids


def parse_keywords(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def analyze_universe(input_df: pd.DataFrame, plan_lookup: dict):
    carrier_to_classifications = {}
    carrier_to_types = {}
    carrier_to_plan_names = {}
    classification_to_carriers = {}
    classification_to_plan_names = {}
    plan_names_universe = set()
    plan_types_universe = set()

    input_plan_ids = extract_input_plan_ids(input_df)
    missing_plan_ids = {pid for pid in input_plan_ids if pid not in plan_lookup}

    for pid in (input_plan_ids - missing_plan_ids):
        info = plan_lookup.get(pid)
        if not info:
            continue

        carrier = normalize_str(info.get("Carrier_Name"))
        pcl = normalize_str(info.get("Plan Classification")) or "(blank)"
        ptype = normalize_str(info.get("Plan_Type")) or "(blank)"
        pname = normalize_str(info.get("Plan_Name")) or "(blank)"

        if not carrier:
            continue

        carrier_to_classifications.setdefault(carrier, set()).add(pcl)
        carrier_to_types.setdefault(carrier, set()).add(ptype)
        carrier_to_plan_names.setdefault(carrier, set()).add(pname)

        classification_to_carriers.setdefault(pcl, set()).add(carrier)
        classification_to_plan_names.setdefault(pcl, set()).add(pname)

        plan_names_universe.add(pname)
        plan_types_universe.add(ptype)

    return {
        "carrier_to_classifications": carrier_to_classifications,
        "carrier_to_types": carrier_to_types,
        "carrier_to_plan_names": carrier_to_plan_names,
        "classification_to_carriers": classification_to_carriers,
        "classification_to_plan_names": classification_to_plan_names,
        "plan_names_universe": plan_names_universe,
        "plan_types_universe": plan_types_universe,
        "missing_plan_ids": missing_plan_ids,
    }


def read_deprecated_plan_ids(dep_df: pd.DataFrame) -> set[str]:
    col = None
    for c in DEPRECATED_ACCEPTABLE_PLAN_ID_COLS:
        if c in dep_df.columns:
            col = c
            break
    if col is None:
        raise ValueError(
            f"Deprecated file must have a Plan_ID column (one of {DEPRECATED_ACCEPTABLE_PLAN_ID_COLS})."
        )

    dep_ids = set()
    for v in dep_df[col].astype(str).fillna("").tolist():
        for pid in split_csv_ids(v):
            dep_ids.add(pid)
    return dep_ids


# -----------------------------
# Performance helpers
# -----------------------------
def build_input_long(input_df: pd.DataFrame) -> pd.DataFrame:
    base = input_df[["MappingLevel", "ProviderId", "LocationId", "PlanIds"]].copy()
    base["PlanId"] = base["PlanIds"].astype(str).apply(split_csv_ids)
    base = base.explode("PlanId").drop(columns=["PlanIds"])
    base["PlanId"] = base["PlanId"].astype(str).map(normalize_str)
    base = base[base["PlanId"] != ""].reset_index(drop=True)
    return base


# -----------------------------
# Classification-level plan-name rule logic
# -----------------------------
def _rule_matches(pname: str, names: set[str], keywords: list[str]) -> bool:
    if pname in names:
        return True
    pname_l = (pname or "").lower()
    for kw in keywords:
        if kw and kw.lower() in pname_l:
            return True
    return False


def compute_removals_fast(
    input_long: pd.DataFrame,
    db_small: pd.DataFrame,
    selected_pairs: set[tuple[str, str]],
    pair_types_map: dict[tuple[str, str], set[str] | None],          # pair -> types; empty=set() => ALL; None => NO MATCH
    pair_plan_names_map: dict[tuple[str, str], set[str]],            # pair -> names; empty=set() => ALL (kept)
    enable_pair_plan_name_filter: bool,
    pair_plan_name_keywords: list[str],
    class_plan_rule_active: dict[str, dict],                         # classification -> active rule
) -> dict[str, pd.DataFrame]:
    if not selected_pairs:
        return {}

    df = input_long.merge(
        db_small,
        left_on="PlanId",
        right_on="Plan_ID",
        how="left",
        copy=False,
    )
    df = df[df["Plan_ID"].notna()].copy()
    if df.empty:
        return {}

    # Normalize
    df["Carrier_Name"] = df["Carrier_Name"].astype(str).map(normalize_str)
    df["Plan_Type"] = df["Plan_Type"].astype(str).map(normalize_str)
    df["Plan Classification"] = df["Plan Classification"].astype(str).map(normalize_str)
    df["Plan_Name"] = df["Plan_Name"].astype(str).map(normalize_str)

    df.loc[df["Plan_Type"] == "", "Plan_Type"] = "(blank)"
    df.loc[df["Plan Classification"] == "", "Plan Classification"] = "(blank)"
    df.loc[df["Plan_Name"] == "", "Plan_Name"] = "(blank)"

    # Filter to selected pairs
    allow_df = pd.DataFrame(list(selected_pairs), columns=["Carrier_Name", "Plan Classification"])
    df = df.merge(allow_df, on=["Carrier_Name", "Plan Classification"], how="inner", copy=False)
    if df.empty:
        return {}

    # Apply ACTIVE classification plan-name rules (ONLY / ALL_EXCEPT)
    if class_plan_rule_active:
        keep_mask = pd.Series(True, index=df.index)

        for pcl, rule in class_plan_rule_active.items():
            if not rule:
                continue
            mode = rule.get("mode", "ALL")
            if mode == "ALL":
                continue

            names = set(rule.get("names", set()) or set())
            kws = [k for k in (rule.get("keywords", []) or []) if k]

            sub_idx = df.index[df["Plan Classification"] == pcl]
            if len(sub_idx) == 0:
                continue

            sub = df.loc[sub_idx]
            name_match = sub["Plan_Name"].isin(names) if names else pd.Series(False, index=sub.index)
            if kws:
                pat = "|".join(re.escape(k) for k in kws)
                kw_match = sub["Plan_Name"].str.contains(pat, case=False, na=False, regex=True)
            else:
                kw_match = pd.Series(False, index=sub.index)

            m = name_match | kw_match

            if mode == "ONLY":
                keep_mask.loc[sub_idx] = m.values
            elif mode == "ALL_EXCEPT":
                keep_mask.loc[sub_idx] = (~m).values

        df = df[keep_mask].copy()
        if df.empty:
            return {}

    # Remove NO MATCH pairs
    no_match_pairs = {k for k, v in pair_types_map.items() if v is None}
    if no_match_pairs:
        tmp = pd.DataFrame(list(no_match_pairs), columns=["Carrier_Name", "Plan Classification"])
        df = df.merge(tmp.assign(__drop=1), on=["Carrier_Name", "Plan Classification"], how="left")
        df = df[df["__drop"].isna()].drop(columns=["__drop"])
        if df.empty:
            return {}

    # Types restrictions per pair
    restricted_types = {k: v for k, v in pair_types_map.items() if isinstance(v, set) and len(v) > 0}
    if restricted_types:
        allow_rows = []
        for (carrier, cls), typeset in restricted_types.items():
            for t in typeset:
                allow_rows.append((carrier, cls, t))
        allow_types_df = pd.DataFrame(allow_rows, columns=["Carrier_Name", "Plan Classification", "Plan_Type"])

        restricted_pairs_df = pd.DataFrame(list(restricted_types.keys()), columns=["Carrier_Name", "Plan Classification"])
        df_restr = df.merge(restricted_pairs_df, on=["Carrier_Name", "Plan Classification"], how="inner")
        df_all = df.merge(restricted_pairs_df.assign(__r=1), on=["Carrier_Name", "Plan Classification"], how="left")
        df_all = df_all[df_all["__r"].isna()].drop(columns=["__r"])

        df_restr = df_restr.merge(allow_types_df, on=["Carrier_Name", "Plan Classification", "Plan_Type"], how="inner")
        df = pd.concat([df_all, df_restr], ignore_index=True)
        if df.empty:
            return {}

    # Existing per-pair plan-name filter (kept)
    do_pair_name_filter = False
    if enable_pair_plan_name_filter:
        any_explicit = any(isinstance(v, set) and len(v) > 0 for v in pair_plan_names_map.values())
        any_kw = len([k for k in pair_plan_name_keywords if k]) > 0
        do_pair_name_filter = any_explicit or any_kw

    if do_pair_name_filter:
        if pair_plan_name_keywords:
            kws = [k for k in pair_plan_name_keywords if k]
            if kws:
                pat = "|".join(re.escape(k) for k in kws)
                kw_mask = df["Plan_Name"].str.contains(pat, case=False, na=False, regex=True)
            else:
                kw_mask = pd.Series(False, index=df.index)
        else:
            kw_mask = pd.Series(False, index=df.index)

        restricted_names = {k: v for k, v in pair_plan_names_map.items() if isinstance(v, set) and len(v) > 0}
        name_mask = pd.Series(False, index=df.index)

        if restricted_names:
            allow_rows = []
            for (carrier, cls), nameset in restricted_names.items():
                for n in nameset:
                    allow_rows.append((carrier, cls, n))
            allow_names_df = pd.DataFrame(allow_rows, columns=["Carrier_Name", "Plan Classification", "Plan_Name"])

            tmp = df[["Carrier_Name", "Plan Classification", "Plan_Name"]].copy()
            tmp["_idx"] = tmp.index
            matched = tmp.merge(allow_names_df, on=["Carrier_Name", "Plan Classification", "Plan_Name"], how="inner")
            if not matched.empty:
                name_mask.loc[matched["_idx"].values] = True

        pairs_all_names = {k for k, v in pair_plan_names_map.items() if isinstance(v, set) and len(v) == 0}
        restricted_pairs = set(restricted_names.keys()) if restricted_names else set()

        pair_key_series = list(zip(df["Carrier_Name"], df["Plan Classification"]))
        pair_all_mask = pd.Series(
            [(pk in pairs_all_names) or (pk not in restricted_pairs) for pk in pair_key_series],
            index=df.index,
        )

        keep = (kw_mask | name_mask) if pair_plan_name_keywords else (pair_all_mask | name_mask)
        df = df[keep].copy()
        if df.empty:
            return {}

    out_df = df[["MappingLevel", "ProviderId", "LocationId", "PlanId"]].drop_duplicates(
        subset=["MappingLevel", "ProviderId", "LocationId", "PlanId"]
    )

    tabs = {}
    for lvl, g in out_df.groupby("MappingLevel", sort=False):
        tabs[str(lvl)] = g[["ProviderId", "LocationId", "PlanId"]].reset_index(drop=True)

    return tabs


def make_excel_bytes(tabs: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, df in tabs.items():
            safe_name = (sheet or "UnknownLevel")[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    bio.seek(0)
    return bio.getvalue()


# -----------------------------
# Session state init
# -----------------------------
if "selected_pairs" not in st.session_state:
    st.session_state.selected_pairs = set()  # {(carrier, classification)}

if "pair_types_map" not in st.session_state:
    st.session_state.pair_types_map = {}

if "pair_plan_names_map" not in st.session_state:
    st.session_state.pair_plan_names_map = {}

# ACTIVE rules only apply after user clicks Apply/Confirm
if "class_plan_rule_active" not in st.session_state:
    st.session_state.class_plan_rule_active = {}

if "carrier_widget_value" not in st.session_state:
    st.session_state.carrier_widget_value = []

if "active_global_class_filter" not in st.session_state:
    st.session_state.active_global_class_filter = None

if "prev_class_filter" not in st.session_state:
    st.session_state.prev_class_filter = None

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "bulk_default_types" not in st.session_state:
    st.session_state.bulk_default_types = []

# Finder cache for UI
if "finder_results" not in st.session_state:
    st.session_state.finder_results = []


# -----------------------------
# UI: Header + Notes
# -----------------------------
st.title("Removal Tool")

st.info(
    "📌 **DB file requirements (upload .xlsx or .csv):**\n"
    f"- Must contain these columns exactly: **{', '.join(DB_REQUIRED_COLS)}**\n"
    "- **Plan_ID** values should include the `ip_` prefix (example: `ip_20356`).\n"
    "\n"
    "📌 **Input CSV requirements:**\n"
    f"- Must contain: **{', '.join(INPUT_REQUIRED_COLS)}**\n"
)


# -----------------------------
# Uploads
# -----------------------------
col_up1, col_up2, col_up3 = st.columns([1, 1, 1])

with col_up1:
    input_file = st.file_uploader("Upload Input CSV", type=["csv"])

with col_up2:
    db_file = st.file_uploader("Upload DB (Excel/CSV)", type=["xlsx", "xls", "csv"])

with col_up3:
    dep_file = st.file_uploader("Upload Deprecated Plans (optional)", type=["xlsx", "xls", "csv"])

load_btn = st.button("Load & Analyze", type="primary")


# -----------------------------
# Load & Analyze
# -----------------------------
if load_btn:
    if input_file is None or db_file is None:
        st.error("Please upload both Input CSV and DB file.")
    else:
        prog = st.progress(0, text="Starting...")
        try:
            prog.progress(5, text="Reading input file...")
            input_df = read_table(input_file)
            validate_columns(input_df, INPUT_REQUIRED_COLS, "Input CSV")
            for c in INPUT_REQUIRED_COLS:
                input_df[c] = input_df[c].astype(str).map(normalize_str)

            prog.progress(20, text="Reading DB file...")
            db_df = read_table(db_file)
            validate_columns(db_df, DB_REQUIRED_COLS, "DB file")

            prog.progress(35, text="Normalizing DB Plan_ID values...")
            db_df = explode_db_plan_ids(db_df)

            prog.progress(55, text="Building plan lookup...")
            plan_lookup = build_plan_lookup(db_df)

            prog.progress(70, text="Analyzing carriers/classifications/types/plan names...")
            universe = analyze_universe(input_df, plan_lookup)

            missing_plan_ids = universe["missing_plan_ids"]

            deprecated_found = 0
            deprecated_total = 0
            if dep_file is not None:
                prog.progress(80, text="Reading deprecated file...")
                dep_df = read_table(dep_file)
                dep_ids = read_deprecated_plan_ids(dep_df)
                deprecated_total = len(dep_ids)
                deprecated_found = len(missing_plan_ids.intersection(dep_ids))

            prog.progress(88, text="Preparing performance caches...")
            st.session_state.input_long = build_input_long(input_df)
            st.session_state.db_small = db_df[
                ["Plan_ID", "Carrier_Name", "Plan_Type", "Plan Classification", "Plan_Name"]
            ].copy()

            prog.progress(94, text="Preparing UI data...")
            all_carriers = sorted(universe["carrier_to_classifications"].keys(), key=lambda x: x.lower())
            all_classifications = sorted(universe["classification_to_carriers"].keys(), key=lambda x: x.lower())
            all_types = sorted(universe["plan_types_universe"], key=lambda x: x.lower())

            st.session_state.input_df = input_df
            st.session_state.db_df = db_df
            st.session_state.plan_lookup = plan_lookup
            st.session_state.universe = universe

            st.session_state.all_carriers = all_carriers
            st.session_state.all_classifications = all_classifications
            st.session_state.all_types = all_types
            st.session_state.classification_to_plan_names = universe["classification_to_plan_names"]

            st.session_state.missing_plan_ids_count = len(missing_plan_ids)
            st.session_state.deprecated_found = deprecated_found
            st.session_state.deprecated_total = deprecated_total

            st.session_state.active_global_class_filter = None
            st.session_state.prev_class_filter = None

            # reset selections
            st.session_state.selected_pairs = set()
            st.session_state.pair_types_map = {}
            st.session_state.pair_plan_names_map = {}
            st.session_state.class_plan_rule_active = {}
            st.session_state.carrier_widget_value = []
            st.session_state.finder_results = []

            st.session_state.loaded = True

            prog.progress(100, text="Done.")
            st.success("Loaded and analyzed successfully.")
        except Exception as e:
            prog.progress(100, text="Failed.")
            st.error(str(e))


# -----------------------------
# Main UI
# -----------------------------
if st.session_state.loaded:
    universe = st.session_state.universe
    carrier_to_plan_names = universe["carrier_to_plan_names"]
    classification_to_carriers = universe["classification_to_carriers"]
    carrier_to_types = universe["carrier_to_types"]

    # ---------------------------
    # SIDEBAR: classification + view search
    # ---------------------------
    with st.sidebar:
        st.header("Filters & Search")

        st.caption("Plan Classification (required for carrier picking)")
        class_filter = st.selectbox(
            "Plan Classification",
            options=["(select)"] + st.session_state.all_classifications,
            index=0,
            format_func=lambda x: "Select a classification" if x == "(select)" else x,
        )

        st.divider()
        st.caption("🔎 View-only search (does NOT change your selection)")
        st.write("This only changes what you SEE on the left. It will never remove existing summary selections.")
        carrier_search = st.text_input(
            "Search carriers OR plan names",
            value="",
            placeholder="Example: cigna OR rae OR medicaid",
        )

        st.divider()
        st.caption("Row-level plan-name keywords (optional, applies during removal)")
        kw_text = st.text_area("Keywords (comma/newline)", value="", height=80)
        pair_plan_name_keywords = parse_keywords(kw_text)
        enable_pair_plan_name_filter = True

        st.divider()
        miss = st.session_state.missing_plan_ids_count
        dep_found = st.session_state.get("deprecated_found", 0)
        if st.session_state.get("deprecated_total", 0):
            st.warning(f"Missing in DB: {miss} | Deprecated matched: {dep_found}")
        else:
            st.info(f"Missing in DB: {miss} (upload deprecated file to match)")

    # Reset left carrier picker when class changes (blank picker), but DO NOT touch selections
    if class_filter != st.session_state.prev_class_filter:
        st.session_state.carrier_widget_value = []
        st.session_state.finder_results = []
        st.session_state.prev_class_filter = class_filter

    # displayed carriers from chosen classification only
    if class_filter != "(select)":
        st.session_state.active_global_class_filter = class_filter
        carriers_base = sorted(list(classification_to_carriers.get(class_filter, set())), key=lambda x: x.lower())
    else:
        st.session_state.active_global_class_filter = None
        carriers_base = []

    # View-only search: filter visible list, never changes selection
    if carriers_base and carrier_search.strip():
        q = carrier_search.strip().lower()
        filtered = []
        for c in carriers_base:
            if q in c.lower():
                filtered.append(c)
                continue
            pnames = carrier_to_plan_names.get(c, set())
            if any(q in (pn or "").lower() for pn in pnames):
                filtered.append(c)
        displayed_carriers = sorted(filtered, key=lambda x: x.lower())
    else:
        displayed_carriers = carriers_base

    st.subheader("Dashboard")
    left, right = st.columns([2, 1], gap="large")

    # ---------------------------
    # LEFT: Carrier selection (ADD-only; never auto-removes)
    # ---------------------------
    with left:
        st.markdown("### 1) Pick Carriers")

        active_cls = st.session_state.active_global_class_filter
        disabled_pick = active_cls is None

        if disabled_pick:
            st.info("Select a Plan Classification in the left sidebar to load carriers.")
        else:
            glimpse_n = min(5, len(displayed_carriers))
            if glimpse_n > 0:
                st.caption("Glimpse: " + " • ".join(displayed_carriers[:glimpse_n]))
            else:
                st.caption("No carriers found for this classification (after search filter).")

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            add_all_shown = st.button("Select all shown", disabled=disabled_pick)
        with colB:
            clear_all = st.button("Clear all")
        with colC:
            selected_total = len(st.session_state.selected_pairs)
            selected_in_cls = (
                len([1 for (_, cls) in st.session_state.selected_pairs if cls == active_cls])
                if active_cls is not None
                else 0
            )
            st.caption(
                f"Shown: {len(displayed_carriers)} | Selected (this class): {selected_in_cls} | Selected (total): {selected_total}"
            )

        if clear_all:
            st.session_state.selected_pairs = set()
            st.session_state.pair_types_map = {}
            st.session_state.pair_plan_names_map = {}
            st.session_state.carrier_widget_value = []
            st.session_state.finder_results = []

        # ADDITIVE: Select all shown just adds pairs; never removes anything
        if add_all_shown and active_cls is not None:
            for carrier in displayed_carriers:
                pair = (carrier, active_cls)
                st.session_state.selected_pairs.add(pair)
                st.session_state.pair_types_map.setdefault(pair, set())       # empty => ALL
                st.session_state.pair_plan_names_map.setdefault(pair, set()) # empty => ALL
            st.success(f"Added {len(displayed_carriers)} carriers for {active_cls} (selection preserved).")

        # Multiselect is now "pick some to add" (not source of truth)
        picked_to_add = st.multiselect(
            "Select carriers (then click Add selected)",
            options=displayed_carriers,
            key="carrier_widget_value",
            label_visibility="collapsed",
            disabled=disabled_pick,
        )
        if st.button("Add selected carriers", use_container_width=True, disabled=disabled_pick or (not picked_to_add)):
            for carrier in picked_to_add:
                pair = (carrier, active_cls)
                st.session_state.selected_pairs.add(pair)
                st.session_state.pair_types_map.setdefault(pair, set())
                st.session_state.pair_plan_names_map.setdefault(pair, set())
            st.session_state.carrier_widget_value = []  # clear picker
            st.success(f"Added {len(picked_to_add)} carriers for {active_cls} (selection preserved).")

        # ---- Finder: plan-name keyword -> carriers (ADD results without touching existing) ----
        st.divider()
        st.markdown("### Finder: carriers that have plan-name keyword (additive)")
        st.caption("Use this to discover carriers. Adding results will never remove your existing selection.")

        finder_kw = st.text_input(
            "Find carriers by plan-name keyword",
            value="",
            placeholder="Example: rae OR rocky OR denver",
            disabled=disabled_pick,
            key="finder_kw",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            run_finder = st.button("Find carriers", disabled=disabled_pick or not finder_kw.strip(), use_container_width=True)
        with c2:
            add_finder = st.button(
                "Add finder results",
                disabled=disabled_pick or (len(st.session_state.finder_results) == 0),
                use_container_width=True,
            )

        if run_finder and active_cls is not None:
            kw = finder_kw.strip().lower()
            results = []
            # Search carriers in this classification for plan-name keyword
            for c in sorted(list(classification_to_carriers.get(active_cls, set())), key=lambda x: x.lower()):
                pnames = carrier_to_plan_names.get(c, set())
                if any(kw in (pn or "").lower() for pn in pnames):
                    results.append(c)
            st.session_state.finder_results = results
            st.info(f"Finder results: {len(results)} carriers found for '{finder_kw}' under {active_cls}.")

        if st.session_state.finder_results:
            st.write("Found carriers:")
            st.write(", ".join(st.session_state.finder_results[:30]) + (" ..." if len(st.session_state.finder_results) > 30 else ""))

        if add_finder and active_cls is not None:
            added = 0
            for carrier in st.session_state.finder_results:
                pair = (carrier, active_cls)
                if pair not in st.session_state.selected_pairs:
                    added += 1
                st.session_state.selected_pairs.add(pair)
                st.session_state.pair_types_map.setdefault(pair, set())
                st.session_state.pair_plan_names_map.setdefault(pair, set())
            st.success(f"Added {added} new carriers from finder for {active_cls} (selection preserved).")

        st.markdown("### Tips")
        st.caption(
            "- Search boxes only change what you SEE.\n"
            "- Buttons ADD carriers; they never remove your existing selection.\n"
            "- To remove something, use the right-side Summary controls."
        )

    # ---------------------------
    # RIGHT: Summary + apply/confirm plan-name rules + preview/generate
    # ---------------------------
    with right:
        st.markdown("### Summary")

        selected_pairs_sorted = sorted(list(st.session_state.selected_pairs), key=lambda x: (x[0].lower(), x[1].lower()))
        if not selected_pairs_sorted:
            st.info("Select carriers to enable preview + output.")
        else:
            # Summary table
            rows = []
            for pair in selected_pairs_sorted:
                carrier, cls = pair
                types_val = st.session_state.pair_types_map.get(pair, set())

                if types_val is None:
                    types_label = "NO MATCH (0 rows)"
                elif not types_val:
                    types_label = "ALL"
                else:
                    types_label = f"{len(types_val)} selected"

                rows.append(
                    {
                        "Carrier": carrier,
                        "Plan Classification": cls,
                        "Plan Types": types_label,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Remove selected rows explicitly (safe removal)
            st.divider()
            st.markdown("### Remove selected rows (explicit)")
            to_remove_labels = [f"{c} | {cls}" for (c, cls) in selected_pairs_sorted]
            pick_remove = st.multiselect(
                "Choose rows to remove from summary",
                options=to_remove_labels,
                default=[],
                key="rows_to_remove",
            )
            if st.button("Remove chosen rows", use_container_width=True, disabled=not pick_remove):
                remove_set = set()
                for lab in pick_remove:
                    c, cls = [p.strip() for p in lab.split("|", 1)]
                    remove_set.add((c, cls))

                st.session_state.selected_pairs -= remove_set
                for pair in remove_set:
                    st.session_state.pair_types_map.pop(pair, None)
                    st.session_state.pair_plan_names_map.pop(pair, None)

                st.success(f"Removed {len(remove_set)} rows from summary.")
                st.session_state.rows_to_remove = []

            # Bulk types apply
            st.divider()
            with st.expander("Bulk apply plan types to ALL selected rows", expanded=False):
                bulk_types = st.multiselect(
                    "Default Plan Types (leave empty = ALL)",
                    options=st.session_state.all_types,
                    default=st.session_state.bulk_default_types,
                    key="bulk_default_types",
                )

                if st.button("Apply these Plan Types to ALL selected rows", use_container_width=True):
                    requested = set(bulk_types)
                    no_match_pairs = []

                    for (carrier, cls) in selected_pairs_sorted:
                        allowed = carrier_to_types.get(carrier, set())
                        valid = requested.intersection(allowed)

                        if requested and not valid:
                            st.session_state.pair_types_map[(carrier, cls)] = None
                            no_match_pairs.append((carrier, cls))
                        else:
                            st.session_state.pair_types_map[(carrier, cls)] = set(valid)

                    if no_match_pairs:
                        st.warning(
                            f"{len(no_match_pairs)} rows had **no matching plan types** for your bulk selection "
                            f"(marked as **NO MATCH**; they will remove 0 rows unless edited)."
                        )
                    else:
                        st.success("Applied plan types to all selected rows.")

            # ---- Classification plan-name rules with Apply/Confirm ----
            st.divider()
            st.markdown("### Plan name rules (by classification)")

            active_cls = st.session_state.active_global_class_filter
            if active_cls is None:
                st.info("Select a Plan Classification on the left to configure plan-name rules for it.")
            else:
                active_rule = st.session_state.class_plan_rule_active.get(active_cls, {"mode": "ALL", "names": set(), "keywords": []})

                # Show active rule banner
                if active_rule.get("mode", "ALL") == "ALL":
                    st.caption(f"Active rule for **{active_cls}**: None (remove all plans).")
                elif active_rule.get("mode") == "ONLY":
                    st.success(
                        f"Active rule for **{active_cls}**: Remove ONLY selected plan names/keywords "
                        f"({len(active_rule.get('names', set()) or set())} names, {len(active_rule.get('keywords', []) or [])} keywords)."
                    )
                elif active_rule.get("mode") == "ALL_EXCEPT":
                    st.success(
                        f"Active rule for **{active_cls}**: Remove ALL EXCEPT selected plan names/keywords "
                        f"({len(active_rule.get('names', set()) or set())} names, {len(active_rule.get('keywords', []) or [])} keywords)."
                    )

                # Draft UI (does not apply until you click Apply)
                st.caption("Draft changes below will NOT take effect until you click **Apply / Confirm**.")

                draft_mode = st.radio(
                    f"Draft mode for {active_cls}",
                    options=["ALL", "ONLY", "ALL_EXCEPT"],
                    index={"ALL": 0, "ONLY": 1, "ALL_EXCEPT": 2}.get(active_rule.get("mode", "ALL"), 0),
                    format_func=lambda x: {
                        "ALL": "Remove all plans (no plan-name rule)",
                        "ONLY": "Remove ONLY selected plan names / keywords",
                        "ALL_EXCEPT": "Remove ALL EXCEPT selected plan names / keywords (preserve list)",
                    }[x],
                    key=f"draft_mode__{active_cls}",
                )

                all_names = sorted(
                    list(st.session_state.classification_to_plan_names.get(active_cls, set())),
                    key=lambda x: (x or "").lower(),
                )
                q = st.text_input("Search plan names", value="", key=f"draft_plan_search__{active_cls}")
                visible = [n for n in all_names if q.strip().lower() in (n or "").lower()] if q.strip() else all_names

                # draft selected names default from ACTIVE
                active_names = set(active_rule.get("names", set()) or set())
                default_visible = sorted(list(active_names.intersection(set(visible))), key=lambda x: (x or "").lower())

                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Draft: select all shown", key=f"draft_selall__{active_cls}", use_container_width=True):
                        # store in session as helper
                        st.session_state[f"draft_names_cache__{active_cls}"] = list(set(visible) | active_names)
                with c2:
                    if st.button("Draft: clear", key=f"draft_clear__{active_cls}", use_container_width=True):
                        st.session_state[f"draft_names_cache__{active_cls}"] = []
                with c3:
                    clear_rule = st.button("Clear ACTIVE rule", key=f"clear_active_rule__{active_cls}", use_container_width=True)

                cache_key = f"draft_names_cache__{active_cls}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = list(active_names)

                # if user used buttons, those values are the defaults
                cached = st.session_state[cache_key]
                cached_visible = sorted(list(set(cached).intersection(set(visible))), key=lambda x: (x or "").lower())

                draft_names = st.multiselect(
                    "Draft selected plan names",
                    options=visible,
                    default=cached_visible if cached_visible else default_visible,
                    key=f"draft_names__{active_cls}",
                )

                # keep hidden draft names across searches
                keep_hidden = set(st.session_state[cache_key]) - set(visible)
                final_draft_names = keep_hidden | set(draft_names)
                st.session_state[cache_key] = list(final_draft_names)

                draft_kw_text = st.text_area(
                    "Draft keywords (comma/newline)",
                    value="\n".join(active_rule.get("keywords", []) or []),
                    height=70,
                    key=f"draft_keywords__{active_cls}",
                )
                draft_keywords = parse_keywords(draft_kw_text)

                col_apply1, col_apply2 = st.columns(2)
                with col_apply1:
                    apply_rule = st.button("Apply / Confirm draft rule", use_container_width=True, key=f"apply_rule__{active_cls}")
                with col_apply2:
                    noop = st.button("Do nothing", use_container_width=True, key=f"noop__{active_cls}")

                if clear_rule:
                    st.session_state.class_plan_rule_active.pop(active_cls, None)
                    st.success(f"Cleared ACTIVE rule for {active_cls} (back to remove all).")

                if apply_rule:
                    st.session_state.class_plan_rule_active[active_cls] = {
                        "mode": draft_mode,
                        "names": set(final_draft_names),
                        "keywords": list(draft_keywords),
                    }
                    st.success(f"Applied ACTIVE rule for {active_cls}.")

            # ---- Preview counts (FAST) ----
            st.divider()
            if st.button("Preview removal counts", use_container_width=True):
                with st.spinner("Building preview (optimized)..."):
                    tabs_preview = compute_removals_fast(
                        input_long=st.session_state.input_long,
                        db_small=st.session_state.db_small,
                        selected_pairs=set(st.session_state.selected_pairs),
                        pair_types_map=dict(st.session_state.pair_types_map),
                        pair_plan_names_map=dict(st.session_state.pair_plan_names_map),
                        enable_pair_plan_name_filter=enable_pair_plan_name_filter,
                        pair_plan_name_keywords=list(pair_plan_name_keywords),
                        class_plan_rule_active=dict(st.session_state.class_plan_rule_active),
                    )

                preview_rows = [{"MappingLevel": k, "Rows": len(v)} for k, v in tabs_preview.items()]
                preview_df = (
                    pd.DataFrame(preview_rows).sort_values("MappingLevel")
                    if preview_rows
                    else pd.DataFrame(columns=["MappingLevel", "Rows"])
                )
                total_preview = int(preview_df["Rows"].sum()) if not preview_df.empty else 0
                st.success(f"Preview ready. Total rows: {total_preview}")
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

            # ---- Final output (FAST) ----
            st.divider()
            generate = st.button("FINAL: Generate Removal Output", type="primary", use_container_width=True)
            if generate:
                with st.spinner("Computing removals (optimized)..."):
                    tabs_final = compute_removals_fast(
                        input_long=st.session_state.input_long,
                        db_small=st.session_state.db_small,
                        selected_pairs=set(st.session_state.selected_pairs),
                        pair_types_map=dict(st.session_state.pair_types_map),
                        pair_plan_names_map=dict(st.session_state.pair_plan_names_map),
                        enable_pair_plan_name_filter=enable_pair_plan_name_filter,
                        pair_plan_name_keywords=list(pair_plan_name_keywords),
                        class_plan_rule_active=dict(st.session_state.class_plan_rule_active),
                    )

                total_rows = sum(len(df) for df in tabs_final.values())
                if total_rows == 0:
                    st.warning("No removals matched.")
                else:
                    st.success(f"Removals: {total_rows} rows | Tabs: {len(tabs_final)}")
                    xbytes = make_excel_bytes(tabs_final)
                    st.download_button(
                        "Download removal_output.xlsx",
                        data=xbytes,
                        file_name="removal_output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
