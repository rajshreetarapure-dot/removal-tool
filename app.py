import io
import re
import hashlib
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
# Rule logic: per (classification, carrier) OR wildcard carrier "*"
# Modes:
#   - ALL        : no rule (remove all)
#   - ONLY       : remove ONLY selected names/keywords
#   - ALL_EXCEPT : remove EVERYTHING EXCEPT selected names/keywords (preserve list)
# -----------------------------
def _mk_rule_key(cls: str, carrier: str) -> tuple[str, str]:
    return (cls, carrier)


def _rule_match_series(plan_name_series: pd.Series, names: set[str], keywords: list[str]) -> pd.Series:
    name_match = plan_name_series.isin(names) if names else pd.Series(False, index=plan_name_series.index)
    if keywords:
        pat = "|".join(re.escape(k) for k in keywords if k)
        if pat:
            kw_match = plan_name_series.str.contains(pat, case=False, na=False, regex=True)
        else:
            kw_match = pd.Series(False, index=plan_name_series.index)
    else:
        kw_match = pd.Series(False, index=plan_name_series.index)

    return name_match | kw_match


def compute_removals_fast(
    input_long: pd.DataFrame,
    db_small: pd.DataFrame,
    selected_pairs: set[tuple[str, str]],
    pair_types_map: dict[tuple[str, str], set[str] | None],   # empty=set() => ALL; None => NO MATCH
    active_plan_rules: dict[tuple[str, str], dict],          # (cls, carrier or "*") -> {"mode","names","keywords"}
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

    # Apply plan rules (specific carrier overrides wildcard)
    if active_plan_rules:
        # Determine which rule applies for each row
        def choose_rule_key(row):
            cls = row["Plan Classification"]
            car = row["Carrier_Name"]
            k_specific = (cls, car)
            k_wild = (cls, "*")
            if k_specific in active_plan_rules:
                return k_specific
            if k_wild in active_plan_rules:
                return k_wild
            return None

        # This is a bit of Python work but OK for typical file sizes
        rule_keys = [choose_rule_key(r) for _, r in df[["Plan Classification", "Carrier_Name"]].iterrows()]
        df["_rule_key"] = rule_keys

        keep_mask = pd.Series(True, index=df.index)

        # Apply each rule to its subset
        for rk in sorted({k for k in df["_rule_key"].unique() if k is not None}):
            rule = active_plan_rules.get(rk)
            if not rule:
                continue
            mode = rule.get("mode", "ALL")
            if mode == "ALL":
                continue

            names = set(rule.get("names", set()) or set())
            kws = list(rule.get("keywords", []) or [])

            sub_idx = df.index[df["_rule_key"] == rk]
            if len(sub_idx) == 0:
                continue

            sub = df.loc[sub_idx]
            m = _rule_match_series(sub["Plan_Name"], names, kws)

            if mode == "ONLY":
                keep_mask.loc[sub_idx] = m.values
            elif mode == "ALL_EXCEPT":
                keep_mask.loc[sub_idx] = (~m).values

        df = df[keep_mask].copy()
        df = df.drop(columns=["_rule_key"])
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

    # Plan type restrictions per pair
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

    # Output tabs
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


def stable_key(*parts: str) -> str:
    raw = "||".join([p or "" for p in parts])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# -----------------------------
# Session state init
# -----------------------------
if "selected_pairs" not in st.session_state:
    st.session_state.selected_pairs = set()  # {(carrier, classification)}

if "pair_types_map" not in st.session_state:
    st.session_state.pair_types_map = {}     # (carrier, cls) -> set(types) or None

if "active_plan_rules" not in st.session_state:
    st.session_state.active_plan_rules = {}  # (cls, carrier or "*") -> {"mode","names","keywords"}

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "active_global_class_filter" not in st.session_state:
    st.session_state.active_global_class_filter = None

if "prev_class_filter" not in st.session_state:
    st.session_state.prev_class_filter = None

if "carrier_add_picker" not in st.session_state:
    st.session_state.carrier_add_picker = []  # widget key safe

if "plan_explorer_selected" not in st.session_state:
    # (cls, carrier) -> set(plan_names checked in explorer)
    st.session_state.plan_explorer_selected = {}

if "plan_explorer_last_results" not in st.session_state:
    # cached results for current cls & keyword: list of (carrier, plan_name)
    st.session_state.plan_explorer_last_results = []

if "bulk_default_types" not in st.session_state:
    st.session_state.bulk_default_types = []


# -----------------------------
# UI: Header
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
            st.session_state.universe = universe
            st.session_state.all_classifications = sorted(universe["classification_to_carriers"].keys(), key=lambda x: x.lower())
            st.session_state.all_types = sorted(universe["plan_types_universe"], key=lambda x: x.lower())
            st.session_state.missing_plan_ids_count = len(missing_plan_ids)
            st.session_state.deprecated_found = deprecated_found
            st.session_state.deprecated_total = deprecated_total

            # Reset state on load
            st.session_state.selected_pairs = set()
            st.session_state.pair_types_map = {}
            st.session_state.active_plan_rules = {}
            st.session_state.active_global_class_filter = None
            st.session_state.prev_class_filter = None
            st.session_state.carrier_add_picker = []
            st.session_state.plan_explorer_selected = {}
            st.session_state.plan_explorer_last_results = []

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
    classification_to_carriers = universe["classification_to_carriers"]
    carrier_to_plan_names = universe["carrier_to_plan_names"]
    carrier_to_types = universe["carrier_to_types"]

    # ---------------------------
    # Sidebar: classification + view-only search
    # ---------------------------
    with st.sidebar:
        st.header("Pick a Classification")

        class_filter = st.selectbox(
            "Plan Classification",
            options=["(select)"] + st.session_state.all_classifications,
            index=0,
            format_func=lambda x: "Select a classification" if x == "(select)" else x,
            key="sidebar_class_filter",
        )

        st.divider()
        st.caption("🔎 View-only search (never removes selection)")
        view_search = st.text_input(
            "Search carriers OR plan names",
            value="",
            placeholder="Example: aetna OR rae OR rocky",
            key="sidebar_view_search",
        )

        st.divider()
        miss = st.session_state.missing_plan_ids_count
        dep_found = st.session_state.get("deprecated_found", 0)
        if st.session_state.get("deprecated_total", 0):
            st.warning(f"Missing in DB: {miss} | Deprecated matched: {dep_found}")
        else:
            st.info(f"Missing in DB: {miss} (upload deprecated file to match)")

    # update active filter
    if class_filter != "(select)":
        st.session_state.active_global_class_filter = class_filter
    else:
        st.session_state.active_global_class_filter = None

    # when class changes, clear view pickers but KEEP summary selections
    if class_filter != st.session_state.prev_class_filter:
        st.session_state.carrier_add_picker = []
        st.session_state.plan_explorer_last_results = []
        st.session_state.prev_class_filter = class_filter

    active_cls = st.session_state.active_global_class_filter
    disabled_pick = active_cls is None

    # compute displayed carriers for this classification (view-only search)
    carriers_base = sorted(list(classification_to_carriers.get(active_cls, set())), key=lambda x: x.lower()) if active_cls else []
    if active_cls and view_search.strip():
        q = view_search.strip().lower()
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
    # LEFT: Carrier selection (ADD-only; safe)
    # ---------------------------
    with left:
        st.markdown("### 1) Pick Carriers")

        if disabled_pick:
            st.info("Select a Plan Classification in the sidebar to load carriers.")
        else:
            glimpse = displayed_carriers[:5]
            st.caption("Glimpse: " + (" • ".join(glimpse) if glimpse else "(none)"))

        def add_all_shown():
            if not st.session_state.active_global_class_filter:
                return
            cls = st.session_state.active_global_class_filter
            shown = st.session_state.get("displayed_carriers_cache", [])
            for carrier in shown:
                pair = (carrier, cls)
                st.session_state.selected_pairs.add(pair)
                st.session_state.pair_types_map.setdefault(pair, set())

        def add_selected_carriers():
            cls = st.session_state.active_global_class_filter
            picked = st.session_state.get("carrier_add_picker", [])
            if not cls or not picked:
                return
            for carrier in picked:
                pair = (carrier, cls)
                st.session_state.selected_pairs.add(pair)
                st.session_state.pair_types_map.setdefault(pair, set())
            st.session_state["carrier_add_picker"] = []

        def clear_all():
            st.session_state.selected_pairs = set()
            st.session_state.pair_types_map = {}
            st.session_state.carrier_add_picker = []

        st.session_state["displayed_carriers_cache"] = list(displayed_carriers)

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            st.button("Select all shown", disabled=disabled_pick, on_click=add_all_shown)
        with colB:
            st.button("Clear ALL selections", on_click=clear_all)
        with colC:
            selected_total = len(st.session_state.selected_pairs)
            selected_in_cls = len([1 for (_, cls) in st.session_state.selected_pairs if cls == active_cls]) if active_cls else 0
            st.caption(f"Shown: {len(displayed_carriers)} | Selected (this class): {selected_in_cls} | Selected (total): {selected_total}")

        st.multiselect(
            "Select carriers to add",
            options=displayed_carriers,
            key="carrier_add_picker",
            label_visibility="collapsed",
            disabled=disabled_pick,
        )
        st.button(
            "Add selected carriers",
            use_container_width=True,
            disabled=disabled_pick,
            on_click=add_selected_carriers,
        )

        st.divider()
        st.markdown("### 2) Plan Name Explorer (smart)")
        st.caption(
            "Type a plan keyword → see carriers + plan names → check what to remove/keep → Confirm.\n"
            "This does NOT change your carrier selection."
        )

        explorer_kw = st.text_input(
            "Plan keyword",
            value="",
            placeholder="Example: rae OR rocky OR denver",
            disabled=disabled_pick,
            key="explorer_kw",
        )

        explorer_mode = st.radio(
            "Checked plans mean:",
            options=["ALLOW (keep these, remove everything else)", "REMOVE ONLY (remove these plans only)"],
            index=0,
            disabled=disabled_pick,
            key="explorer_mode",
        )

        # Build explorer results from db_small
        results = []
        if (not disabled_pick) and explorer_kw.strip():
            kw = explorer_kw.strip().lower()
            dbs = st.session_state.db_small.copy()
            dbs["Carrier_Name"] = dbs["Carrier_Name"].astype(str).map(normalize_str)
            dbs["Plan Classification"] = dbs["Plan Classification"].astype(str).map(normalize_str)
            dbs["Plan_Name"] = dbs["Plan_Name"].astype(str).map(normalize_str)

            dbs = dbs[dbs["Plan Classification"] == active_cls].copy()
            dbs = dbs[dbs["Plan_Name"].str.contains(re.escape(kw), case=False, na=False)].copy()

            if not dbs.empty:
                # distinct carrier-planname pairs
                dbs = dbs[["Carrier_Name", "Plan_Name"]].drop_duplicates()
                results = list(dbs.itertuples(index=False, name=None))

        st.session_state.plan_explorer_last_results = results

        # show results grouped by carrier
        if disabled_pick:
            st.info("Select a classification to use Plan Name Explorer.")
        elif not explorer_kw.strip():
            st.info("Enter a plan keyword to view carriers + plan names.")
        elif not results:
            st.warning("No matching plan names found for this keyword under this classification.")
        else:
            # Group
            by_carrier = {}
            for carrier, pname in results:
                by_carrier.setdefault(carrier, []).append(pname)
            carriers_found = sorted(by_carrier.keys(), key=lambda x: x.lower())

            st.caption(f"Found **{sum(len(v) for v in by_carrier.values())}** matching plan names across **{len(carriers_found)}** carriers.")

            # Bulk actions for visible results
            def bulk_select_all_visible():
                cls = st.session_state.active_global_class_filter
                if not cls:
                    return
                for car in carriers_found:
                    k = (cls, car)
                    existing = st.session_state.plan_explorer_selected.get(k, set())
                    st.session_state.plan_explorer_selected[k] = set(existing) | set(by_carrier.get(car, []))

            def bulk_clear_all_visible():
                cls = st.session_state.active_global_class_filter
                if not cls:
                    return
                for car in carriers_found:
                    k = (cls, car)
                    st.session_state.plan_explorer_selected[k] = set()

            c1, c2 = st.columns(2)
            with c1:
                st.button("Select ALL shown plan names", use_container_width=True, on_click=bulk_select_all_visible)
            with c2:
                st.button("Clear ALL shown checks", use_container_width=True, on_click=bulk_clear_all_visible)

            # Render carrier expanders with checkboxes
            # To avoid too much scrolling, cap carriers shown
            MAX_CARRIERS = 30
            if len(carriers_found) > MAX_CARRIERS:
                st.warning(f"Showing first {MAX_CARRIERS} carriers. Refine keyword to narrow results.")

            carriers_to_render = carriers_found[:MAX_CARRIERS]

            # We will compute new selections for rendered carriers and store them
            for car in carriers_to_render:
                plan_list = sorted(list(set(by_carrier.get(car, []))), key=lambda x: (x or "").lower())
                key_bucket = (active_cls, car)
                current_selected = set(st.session_state.plan_explorer_selected.get(key_bucket, set()) or set())

                with st.expander(f"{car} ({len(plan_list)} matching plans)", expanded=False):
                    new_selected = set(current_selected)

                    for pname in plan_list:
                        ck = f"ck__{stable_key(active_cls, car, pname)}"
                        checked = st.checkbox(
                            pname,
                            value=(pname in current_selected),
                            key=ck,
                        )
                        if checked:
                            new_selected.add(pname)
                        else:
                            new_selected.discard(pname)

                    st.session_state.plan_explorer_selected[key_bucket] = new_selected

        # Confirm / Apply button
        st.divider()

        def apply_explorer_rule():
            cls = st.session_state.active_global_class_filter
            if not cls:
                return

            mode = st.session_state.get("explorer_mode", "ALLOW (keep these, remove everything else)")
            # Apply per carrier based on checked selections
            applied = 0
            for (ccls, car), nameset in st.session_state.plan_explorer_selected.items():
                if ccls != cls:
                    continue
                nameset = set(nameset or set())
                if not nameset:
                    continue

                # Map mode to rule
                if mode.startswith("ALLOW"):
                    rule_mode = "ALL_EXCEPT"  # remove everything except allowed (preserve allow list)
                else:
                    rule_mode = "ONLY"        # remove only selected

                st.session_state.active_plan_rules[(cls, car)] = {
                    "mode": rule_mode,
                    "names": set(nameset),
                    "keywords": [],  # keyword selection already converted to explicit names
                }
                applied += 1

        def clear_rules_for_this_class():
            cls = st.session_state.active_global_class_filter
            if not cls:
                return
            # remove any rule for this class (specific or wildcard)
            to_del = [k for k in st.session_state.active_plan_rules.keys() if k[0] == cls]
            for k in to_del:
                st.session_state.active_plan_rules.pop(k, None)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.button(
                "✅ Confirm / Apply checked plan-name rules",
                use_container_width=True,
                disabled=disabled_pick,
                on_click=apply_explorer_rule,
            )
        with cc2:
            st.button(
                "🧹 Clear ACTIVE plan-name rules for this classification",
                use_container_width=True,
                disabled=disabled_pick,
                on_click=clear_rules_for_this_class,
            )

    # ---------------------------
    # RIGHT: Summary + Active Rules + Preview + Generate
    # ---------------------------
    with right:
        st.markdown("### Summary (what will be processed)")

        selected_pairs_sorted = sorted(list(st.session_state.selected_pairs), key=lambda x: (x[0].lower(), x[1].lower()))
        if not selected_pairs_sorted:
            st.info("Select carriers to enable preview + output.")
        else:
            # Summary table
            rows = []
            for (carrier, cls) in selected_pairs_sorted:
                types_val = st.session_state.pair_types_map.get((carrier, cls), set())
                if types_val is None:
                    types_label = "NO MATCH"
                elif not types_val:
                    types_label = "ALL"
                else:
                    types_label = f"{len(types_val)} selected"
                rows.append({"Carrier": carrier, "Plan Classification": cls, "Plan Types": types_label})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Active rules view
            st.divider()
            st.markdown("### ACTIVE Plan-Name Rules")

            if not st.session_state.active_plan_rules:
                st.caption("No active plan-name rules. (Default = remove all plans under selected carriers/classifications).")
            else:
                # show rules grouped by classification
                rule_rows = []
                for (cls, car), rule in sorted(st.session_state.active_plan_rules.items(), key=lambda x: (x[0][0].lower(), x[0][1].lower())):
                    rule_rows.append({
                        "Classification": cls,
                        "Carrier": car,
                        "Mode": rule.get("mode"),
                        "Plan names count": len(rule.get("names", set()) or set()),
                        "Keywords count": len(rule.get("keywords", []) or []),
                    })
                st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

            # Remove explicitly (never accidental)
            st.divider()
            st.markdown("### Remove selected rows (explicit)")

            labels = [f"{c} | {cls}" for (c, cls) in selected_pairs_sorted]
            remove_pick = st.multiselect("Choose rows to remove", options=labels, default=[], key="remove_pick")
            if st.button("Remove chosen rows", use_container_width=True, disabled=not remove_pick):
                remove_set = set()
                for lab in remove_pick:
                    c, cls = [p.strip() for p in lab.split("|", 1)]
                    remove_set.add((c, cls))

                st.session_state.selected_pairs -= remove_set
                for pair in remove_set:
                    st.session_state.pair_types_map.pop(pair, None)

                st.success(f"Removed {len(remove_set)} rows.")
                st.session_state.remove_pick = []

            # Bulk plan types (optional)
            st.divider()
            with st.expander("Bulk apply plan types to ALL selected rows", expanded=False):
                bulk_types = st.multiselect(
                    "Default Plan Types (leave empty = ALL)",
                    options=st.session_state.all_types,
                    default=st.session_state.bulk_default_types,
                    key="bulk_default_types",
                )

                def apply_bulk_types():
                    requested = set(st.session_state.get("bulk_default_types", []))
                    no_match = 0
                    for (carrier, cls) in st.session_state.selected_pairs:
                        allowed = carrier_to_types.get(carrier, set())
                        valid = requested.intersection(allowed)
                        if requested and not valid:
                            st.session_state.pair_types_map[(carrier, cls)] = None
                            no_match += 1
                        else:
                            st.session_state.pair_types_map[(carrier, cls)] = set(valid)
                    if no_match:
                        st.warning(f"{no_match} rows marked NO MATCH for your type selection.")
                    else:
                        st.success("Applied plan types.")

                st.button("Apply plan types", use_container_width=True, on_click=apply_bulk_types)

            # Preview / Generate
            st.divider()
            if st.button("Preview removal counts", use_container_width=True):
                with st.spinner("Building preview..."):
                    tabs_preview = compute_removals_fast(
                        input_long=st.session_state.input_long,
                        db_small=st.session_state.db_small,
                        selected_pairs=set(st.session_state.selected_pairs),
                        pair_types_map=dict(st.session_state.pair_types_map),
                        active_plan_rules=dict(st.session_state.active_plan_rules),
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

            st.divider()
            generate = st.button("FINAL: Generate Removal Output", type="primary", use_container_width=True)
            if generate:
                with st.spinner("Computing removals..."):
                    tabs_final = compute_removals_fast(
                        input_long=st.session_state.input_long,
                        db_small=st.session_state.db_small,
                        selected_pairs=set(st.session_state.selected_pairs),
                        pair_types_map=dict(st.session_state.pair_types_map),
                        active_plan_rules=dict(st.session_state.active_plan_rules),
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
