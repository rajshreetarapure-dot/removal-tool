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

# Deprecated file: we will look for Plan_ID at minimum (Plan_Name optional)
DEPRECATED_ACCEPTABLE_PLAN_ID_COLS = ["Plan_ID", "PlanId", "plan_id", "planid"]


# -----------------------------
# Helpers (UNCHANGED)
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
    """
    Reads CSV or Excel into DataFrame (all as str).
    """
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
    """
    DB may have comma-separated Plan_ID cells. Explode into one row per Plan_ID.
    """
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
    """
    plan_id -> dict row (first wins)
    """
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


def analyze_universe(input_df: pd.DataFrame, plan_lookup: dict):
    """
    Using only plans that are in input AND found in DB lookup, build:
      - carrier_to_classifications
      - carrier_to_types
      - carrier_to_plan_names
      - classification_to_carriers
      - plan_names_universe
      - plan_types_universe
    Also returns:
      - missing_plan_ids (in input but not in DB)
    """
    carrier_to_classifications = {}
    carrier_to_types = {}
    carrier_to_plan_names = {}
    classification_to_carriers = {}
    plan_names_universe = set()
    plan_types_universe = set()

    input_plan_ids = extract_input_plan_ids(input_df)
    missing_plan_ids = {pid for pid in input_plan_ids if pid not in plan_lookup}

    # walk through input plans that exist in DB
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

        plan_names_universe.add(pname)
        plan_types_universe.add(ptype)

    return {
        "carrier_to_classifications": carrier_to_classifications,
        "carrier_to_types": carrier_to_types,
        "carrier_to_plan_names": carrier_to_plan_names,
        "classification_to_carriers": classification_to_carriers,
        "plan_names_universe": plan_names_universe,
        "plan_types_universe": plan_types_universe,
        "missing_plan_ids": missing_plan_ids,
    }


def compute_removals(
    input_df: pd.DataFrame,
    plan_lookup: dict,
    selected_carriers: set[str],
    carrier_classification_map: dict[str, str | None],  # carrier -> classification or None (None means ALL)
    selected_types_global: set[str],                    # optional global plan type filter
    enable_plan_name_filter: bool,
    selected_plan_names: set[str],
    plan_name_keywords: list[str],                      # keyword contains match
):
    """
    Removal rules:
      - Carrier must be selected
      - If carrier has classification set -> only remove those classification plans for that carrier
      - If carrier classification None -> remove all under that carrier
      - Optional global plan type filter
      - Optional plan name filter:
          - If selected_plan_names: restrict to those plan names
          - If plan_name_keywords: restrict to plan names containing any keyword (case-insensitive)
        If both are provided, either match passes (OR)
    """

    def plan_name_matches(name: str) -> bool:
        if not enable_plan_name_filter:
            return True
        name_l = (name or "").lower()

        # If no name criteria provided, treat as no-op
        if not selected_plan_names and not plan_name_keywords:
            return True

        if name in selected_plan_names:
            return True

        for kw in plan_name_keywords:
            if kw and kw.lower() in name_l:
                return True

        return False

    buckets: dict[str, list[dict]] = {}

    for _, row in input_df.iterrows():
        lvl = normalize_str(row.get("MappingLevel")) or "UnknownLevel"
        provider = normalize_str(row.get("ProviderId"))
        location = normalize_str(row.get("LocationId"))

        for pid in split_csv_ids(row.get("PlanIds", "")):
            info = plan_lookup.get(pid)
            if not info:
                continue

            carrier = normalize_str(info.get("Carrier_Name"))
            ptype = normalize_str(info.get("Plan_Type")) or "(blank)"
            pcl = normalize_str(info.get("Plan Classification")) or "(blank)"
            pname = normalize_str(info.get("Plan_Name")) or "(blank)"

            if carrier not in selected_carriers:
                continue

            if selected_types_global and ptype not in selected_types_global:
                continue

            sel_cls = carrier_classification_map.get(carrier, None)
            if sel_cls is not None and pcl != sel_cls:
                continue

            if not plan_name_matches(pname):
                continue

            buckets.setdefault(lvl, []).append(
                {"ProviderId": provider, "LocationId": location, "PlanId": pid}
            )

    out = {}
    for lvl, rows in buckets.items():
        df = pd.DataFrame(rows).drop_duplicates(subset=["ProviderId", "LocationId", "PlanId"]).reset_index(drop=True)
        out[lvl] = df
    return out


def make_excel_bytes(tabs: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, df in tabs.items():
            safe_name = (sheet or "UnknownLevel")[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    bio.seek(0)
    return bio.getvalue()


def parse_keywords(text: str) -> list[str]:
    """
    Comma/newline separated keywords, stripped.
    """
    if not text:
        return []
    parts = re.split(r"[,\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def read_deprecated_plan_ids(dep_df: pd.DataFrame) -> set[str]:
    # find a plan id column
    col = None
    for c in DEPRECATED_ACCEPTABLE_PLAN_ID_COLS:
        if c in dep_df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"Deprecated file must have a Plan_ID column (one of {DEPRECATED_ACCEPTABLE_PLAN_ID_COLS}).")

    dep_ids = set()
    for v in dep_df[col].astype(str).fillna("").tolist():
        # allow comma separated here too
        for pid in split_csv_ids(v):
            dep_ids.add(pid)
    return dep_ids


# -----------------------------
# Session state init
# -----------------------------
if "selected_carriers" not in st.session_state:
    st.session_state.selected_carriers = set()

if "carrier_classification_map" not in st.session_state:
    st.session_state.carrier_classification_map = {}

# NEW (UI-only): per-carrier plan types (empty set = ALL)
if "carrier_types_map" not in st.session_state:
    st.session_state.carrier_types_map = {}

# NEW (UI-only): stable widget value for carrier multiselect
if "carrier_widget_value" not in st.session_state:
    st.session_state.carrier_widget_value = []

# NEW (UI-only): plan name selection that persists + plan name search
if "selected_plan_names_state" not in st.session_state:
    st.session_state.selected_plan_names_state = set()

if "plan_name_search" not in st.session_state:
    st.session_state.plan_name_search = ""

if "active_global_class_filter" not in st.session_state:
    st.session_state.active_global_class_filter = None

if "loaded" not in st.session_state:
    st.session_state.loaded = False


# -----------------------------
# UI: Header + Notes
# -----------------------------
st.title("Carrier Removal Tool (Streamlit)")

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

            prog.progress(90, text="Preparing UI data...")
            all_carriers = sorted(universe["carrier_to_classifications"].keys(), key=lambda x: x.lower())
            all_classifications = sorted(universe["classification_to_carriers"].keys(), key=lambda x: x.lower())
            all_types = sorted(universe["plan_types_universe"], key=lambda x: x.lower())

            # Save into session_state
            st.session_state.input_df = input_df
            st.session_state.db_df = db_df
            st.session_state.plan_lookup = plan_lookup
            st.session_state.universe = universe

            st.session_state.all_carriers = all_carriers
            st.session_state.all_classifications = all_classifications
            st.session_state.all_types = all_types

            st.session_state.missing_plan_ids_count = len(missing_plan_ids)
            st.session_state.deprecated_found = deprecated_found
            st.session_state.deprecated_total = deprecated_total

            st.session_state.active_global_class_filter = None

            # reset selections
            st.session_state.selected_carriers = set()
            st.session_state.carrier_classification_map = {}
            st.session_state.carrier_types_map = {}
            st.session_state.carrier_widget_value = []
            st.session_state.selected_plan_names_state = set()
            st.session_state.plan_name_search = ""

            st.session_state.loaded = True

            prog.progress(100, text="Done.")
            st.success("Loaded and analyzed successfully.")

        except Exception as e:
            prog.progress(100, text="Failed.")
            st.error(str(e))


# -----------------------------
# Main UI (after loaded)
# -----------------------------
if st.session_state.loaded:
    universe = st.session_state.universe
    carrier_to_plan_names = universe["carrier_to_plan_names"]
    classification_to_carriers = universe["classification_to_carriers"]
    carrier_to_classifications = universe["carrier_to_classifications"]
    carrier_to_types = universe["carrier_to_types"]

    # ---------------------------
    # SIDEBAR: Filters
    # ---------------------------
    with st.sidebar:
        st.header("Controls")

        st.caption("Global classification filter (optional)")
        # UI CHANGE: do not display "(none)"
        class_filter = st.selectbox(
            "Plan Classification",
            options=["(none)"] + st.session_state.all_classifications,
            index=0,
            format_func=lambda x: "All classifications" if x == "(none)" else x,
        )

        st.caption("Search (matches Carrier OR Plan Name)")
        carrier_search = st.text_input("Search", value="")

        st.divider()

        st.caption("Plan Name filter (optional)")
        enable_plan_name_filter = st.checkbox("Enable Plan Name filtering", value=False)

        plan_name_keywords = []
        if enable_plan_name_filter:
            kw_text = st.text_area("Keywords (comma/newline)", value="", height=90)
            plan_name_keywords = parse_keywords(kw_text)
            st.caption("Tip: Use keywords like rae2, rae3 to avoid clicking names.")

        st.divider()

        # Deprecated summary
        miss = st.session_state.missing_plan_ids_count
        dep_found = st.session_state.get("deprecated_found", 0)
        if "deprecated_total" in st.session_state and st.session_state.deprecated_total:
            st.warning(f"Missing in DB: {miss} | Deprecated matched: {dep_found}")
        else:
            st.info(f"Missing in DB: {miss} (upload deprecated file to match)")

    # ---------------------------
    # MAIN: compute displayed_carriers FIRST (prevents NameError / AttributeError)
    # ---------------------------
    if class_filter != "(none)":
        st.session_state.active_global_class_filter = class_filter
        carriers_base = sorted(list(classification_to_carriers.get(class_filter, set())), key=lambda x: x.lower())
    else:
        st.session_state.active_global_class_filter = None
        carriers_base = list(st.session_state.all_carriers)

    if carrier_search.strip():
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
    # LEFT: Carrier selection + Plan-name search/pick (searchable + multi-select)
    # ---------------------------
    with left:
        st.markdown("### Carriers")

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            select_all_shown = st.button("Select all shown")
        with colB:
            clear_all = st.button("Clear all")
        with colC:
            st.caption(f"Shown: {len(displayed_carriers)} | Selected: {len(st.session_state.selected_carriers)}")

        if clear_all:
            st.session_state.selected_carriers = set()
            st.session_state.carrier_classification_map = {}
            st.session_state.carrier_types_map = {}
            st.session_state.carrier_widget_value = []
            st.session_state.selected_plan_names_state = set()

        if select_all_shown:
            newly = set(displayed_carriers) - set(st.session_state.selected_carriers)
            st.session_state.selected_carriers |= set(displayed_carriers)

            active_cls = st.session_state.active_global_class_filter
            for carrier in newly:
                if active_cls is not None:
                    st.session_state.carrier_classification_map[carrier] = active_cls
                else:
                    st.session_state.carrier_classification_map.setdefault(carrier, None)
                st.session_state.carrier_types_map.setdefault(carrier, set())

            # sync widget value to include all displayed carriers selected
            shown = set(displayed_carriers)
            keep_hidden = set(st.session_state.carrier_widget_value) - shown
            st.session_state.carrier_widget_value = sorted(list(keep_hidden | shown), key=lambda x: x.lower())

        # Ensure widget state is always a subset of displayed_carriers + (hidden selections handled separately)
        shown_set = set(displayed_carriers)
        prev_selected = set(st.session_state.selected_carriers)

        # If widget empty but we do have previous selections, initialize from previous (shown only)
        if not st.session_state.carrier_widget_value and prev_selected:
            st.session_state.carrier_widget_value = sorted(list(prev_selected.intersection(shown_set)), key=lambda x: x.lower())

        selected_now = st.multiselect(
            "Select carriers",
            options=displayed_carriers,
            key="carrier_widget_value",
            label_visibility="collapsed",
        )

        # Merge hidden selections so search doesn’t wipe them
        prev = set(st.session_state.selected_carriers)
        picked = set(selected_now)
        keep_hidden = prev - shown_set
        st.session_state.selected_carriers = keep_hidden | picked

        # Apply defaults for newly selected
        newly_selected = st.session_state.selected_carriers - prev
        active_cls = st.session_state.active_global_class_filter
        for carrier in newly_selected:
            if active_cls is not None:
                st.session_state.carrier_classification_map[carrier] = active_cls
            else:
                st.session_state.carrier_classification_map.setdefault(carrier, None)
            st.session_state.carrier_types_map.setdefault(carrier, set())

        # -------- Plan Names: search + multi-select (under selected carriers) --------
        if enable_plan_name_filter and st.session_state.selected_carriers:
            st.markdown("### Plan Names (under selected carriers)")

            st.session_state.plan_name_search = st.text_input(
                "Search plan names",
                value=st.session_state.plan_name_search,
                placeholder="Type to filter plan names (e.g., rae2, rae3)",
                key="plan_name_search_box",
            )

            plan_names_pool = set()
            for carrier in st.session_state.selected_carriers:
                plan_names_pool |= carrier_to_plan_names.get(carrier, set())
            plan_names_pool = sorted(list(plan_names_pool), key=lambda x: x.lower())

            qn = st.session_state.plan_name_search.strip().lower()
            if qn:
                plan_names_pool = [pn for pn in plan_names_pool if qn in (pn or "").lower()]

            prev_names = set(st.session_state.selected_plan_names_state)
            shown_names_set = set(plan_names_pool)
            keep_hidden_names = prev_names - shown_names_set

            picked_names = st.multiselect(
                "Pick plan names (multi-select)",
                options=plan_names_pool,
                default=sorted(list(prev_names.intersection(shown_names_set)), key=lambda x: x.lower()),
                key="plan_names_pick",
            )

            st.session_state.selected_plan_names_state = keep_hidden_names | set(picked_names)
            st.caption(f"Plan names selected: {len(st.session_state.selected_plan_names_state)}")

    # ---------------------------
    # RIGHT: Per-carrier Classification + Plan Types (dropdown + multi-select), Final action
    # ---------------------------
    with right:
        st.markdown("### Summary")

        selected_carriers_sorted = sorted(list(st.session_state.selected_carriers), key=lambda x: x.lower())

        if not selected_carriers_sorted:
            st.info("Select carriers to enable output.")
        else:
            rows = []
            for carrier in selected_carriers_sorted:
                cls_val = st.session_state.carrier_classification_map.get(carrier, None)
                types_val = st.session_state.carrier_types_map.get(carrier, set()) or set()
                rows.append(
                    {
                        "Carrier": carrier,
                        "Plan Classification": cls_val if cls_val is not None else "ALL",
                        "Plan Types": sorted(list(types_val), key=lambda x: x.lower()),
                    }
                )
            df_edit = pd.DataFrame(rows)

            st.caption(
                "Edit per carrier:\n"
                "- **Plan Classification**: `ALL` = remove everything under that carrier.\n"
                "- **Plan Types**: leave empty = ALL plan types for that carrier."
            )

            edited = st.data_editor(
                df_edit,
                use_container_width=True,
                disabled=["Carrier"],
                num_rows="fixed",
                column_config={
                    "Plan Classification": st.column_config.SelectboxColumn(
                        "Plan Classification",
                        options=["ALL"] + st.session_state.all_classifications,
                        required=True,
                        help="ALL removes all classifications for that carrier",
                    ),
                    "Plan Types": st.column_config.MultiselectColumn(
                        "Plan Types",
                        options=st.session_state.all_types,
                        help="Leave empty to include ALL plan types for that carrier",
                    ),
                },
            )

            # Apply edits back
            for _, r in edited.iterrows():
                carrier = str(r["Carrier"]).strip()

                v = str(r["Plan Classification"]).strip()
                if v.upper() == "ALL" or v == "":
                    st.session_state.carrier_classification_map[carrier] = None
                else:
                    allowed_cls = carrier_to_classifications.get(carrier, set())
                    if v in allowed_cls:
                        st.session_state.carrier_classification_map[carrier] = v

                tv = r.get("Plan Types", [])
                tv = tv if isinstance(tv, list) else []
                allowed_types = carrier_to_types.get(carrier, set())
                cleaned = [t for t in tv if t in allowed_types]
                st.session_state.carrier_types_map[carrier] = set(cleaned)

            st.divider()

            generate = st.button("FINAL: Generate Removal Output", type="primary", use_container_width=True)

            if generate:
                with st.spinner("Computing removals..."):
                    merged_tabs: dict[str, pd.DataFrame] = {}

                    for carrier in selected_carriers_sorted:
                        per_carrier_types = st.session_state.carrier_types_map.get(carrier, set())

                        tabs = compute_removals(
                            input_df=st.session_state.input_df,
                            plan_lookup=st.session_state.plan_lookup,
                            selected_carriers={carrier},
                            carrier_classification_map=dict(st.session_state.carrier_classification_map),
                            selected_types_global=set(per_carrier_types),  # empty set => ALL
                            enable_plan_name_filter=enable_plan_name_filter,
                            selected_plan_names=set(st.session_state.selected_plan_names_state),
                            plan_name_keywords=list(plan_name_keywords),
                        )

                        for sheet, df in tabs.items():
                            if sheet not in merged_tabs:
                                merged_tabs[sheet] = df.copy()
                            else:
                                merged_tabs[sheet] = pd.concat([merged_tabs[sheet], df], ignore_index=True)

                    for sheet, df in list(merged_tabs.items()):
                        merged_tabs[sheet] = (
                            df.drop_duplicates(subset=["ProviderId", "LocationId", "PlanId"])
                            .reset_index(drop=True)
                        )

                total_rows = sum(len(df) for df in merged_tabs.values())
                if total_rows == 0:
                    st.warning("No removals matched.")
                else:
                    st.success(f"Removals: {total_rows} rows | Tabs: {len(merged_tabs)}")
                    xbytes = make_excel_bytes(merged_tabs)
                    st.download_button(
                        "Download removal_output.xlsx",
                        data=xbytes,
                        file_name="removal_output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
