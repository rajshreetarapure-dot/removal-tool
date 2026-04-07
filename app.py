import io
import re
import gc
import hashlib
import traceback
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Carrier Removal Tool", layout="wide")

# =========================================================
# Required columns
# =========================================================
INPUT_REQUIRED_COLS = ["PracticeId", "ProviderId", "LocationId", "PlanIds", "MappingLevel"]
DB_REQUIRED_COLS = ["Carrier_ID", "Carrier_Name", "Plan_ID", "Plan_Name", "Plan_Type", "Plan Classification"]
DB_SMALL_COLS = ["Plan_ID", "Carrier_Name", "Plan_Type", "Plan Classification", "Plan_Name"]
DEPRECATED_ACCEPTABLE_PLAN_ID_COLS = ["Plan_ID", "PlanId", "plan_id", "planid"]


# =========================================================
# Helpers
# =========================================================
def split_csv_ids(cell_value):
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


def validate_columns(df, required_cols, label):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def stable_key(*parts):
    raw = "||".join([p or "" for p in parts])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def ms_key_for(cls, carrier):
    return f"ms__{stable_key(cls, carrier)}"


def proc_key_for(cls, carrier):
    return f"proc__{stable_key(cls, carrier)}"


def type_override_mode_key_for(cls, carrier):
    return f"type_override_mode__{stable_key(cls, carrier)}"


def type_override_values_key_for(cls, carrier):
    return f"type_override_values__{stable_key(cls, carrier)}"


def get_pair_type_summary(val):
    if val is None:
        return "No matching plan type"
    if not val:
        return "All plan types"
    return ", ".join(sorted(val, key=lambda x: x.lower()))


# =========================================================
# File readers / preprocessors
# =========================================================
def read_uploaded_table(file_bytes, file_name):
    name = file_name.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        return pd.read_csv(bio, dtype=str).fillna("")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(bio, dtype=str).fillna("")
    raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")


def preprocess_input_df(file_bytes, file_name):
    input_df = read_uploaded_table(file_bytes, file_name)
    validate_columns(input_df, INPUT_REQUIRED_COLS, "Input CSV")

    for c in INPUT_REQUIRED_COLS:
        input_df[c] = input_df[c].astype(str).map(normalize_str)

    return input_df


def preprocess_db_df(file_bytes, file_name):
    db_df = read_uploaded_table(file_bytes, file_name)
    validate_columns(db_df, DB_REQUIRED_COLS, "DB file")

    db_df = db_df[DB_REQUIRED_COLS].copy()
    for c in DB_REQUIRED_COLS:
        db_df[c] = db_df[c].astype(str).map(normalize_str)

    db_df["_plan_split"] = db_df["Plan_ID"].apply(split_csv_ids)
    db_df = db_df.explode("_plan_split").reset_index(drop=True)
    db_df["Plan_ID"] = db_df["_plan_split"].fillna("").astype(str).map(normalize_str)
    db_df = db_df.drop(columns=["_plan_split"])
    db_df = db_df[db_df["Plan_ID"] != ""].reset_index(drop=True)

    return db_df


def preprocess_deprecated_ids(file_bytes, file_name):
    dep_df = read_uploaded_table(file_bytes, file_name)

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


def build_plan_lookup(db_df):
    lookup = {}
    for _, row in db_df.iterrows():
        pid = row["Plan_ID"]
        if pid and pid not in lookup:
            lookup[pid] = row.to_dict()
    return lookup


def build_input_long(input_df):
    base = input_df[["MappingLevel", "ProviderId", "LocationId", "PlanIds"]].copy()
    base["PlanId"] = base["PlanIds"].astype(str).apply(split_csv_ids)
    base = base.explode("PlanId").drop(columns=["PlanIds"])
    base["PlanId"] = base["PlanId"].astype(str).map(normalize_str)
    base = base[base["PlanId"] != ""].reset_index(drop=True)
    return base


def extract_input_plan_ids(input_df):
    plan_ids = set()
    for v in input_df["PlanIds"].astype(str).tolist():
        for pid in split_csv_ids(v):
            plan_ids.add(pid)
    return plan_ids


def analyze_universe_from_frames(input_df, db_df):
    plan_lookup = build_plan_lookup(db_df)

    carrier_to_classifications = {}
    carrier_to_types = {}
    carrier_to_plan_names = {}
    classification_to_carriers = {}
    classification_to_types = {}
    class_carrier_to_types = {}
    plan_types_universe = set()

    input_plan_ids = extract_input_plan_ids(input_df)
    missing_plan_ids = set([pid for pid in input_plan_ids if pid not in plan_lookup])

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
        classification_to_types.setdefault(pcl, set()).add(ptype)
        class_carrier_to_types.setdefault((pcl, carrier), set()).add(ptype)
        plan_types_universe.add(ptype)

    return {
        "carrier_to_classifications": carrier_to_classifications,
        "carrier_to_types": carrier_to_types,
        "carrier_to_plan_names": carrier_to_plan_names,
        "classification_to_carriers": classification_to_carriers,
        "classification_to_types": classification_to_types,
        "class_carrier_to_types": class_carrier_to_types,
        "plan_types_universe": plan_types_universe,
        "missing_plan_ids": missing_plan_ids,
    }


# =========================================================
# Plan-name rules
# =========================================================
def _rule_match_series(plan_name_series, names, keywords):
    name_match = plan_name_series.isin(names) if names else pd.Series(False, index=plan_name_series.index)
    if keywords:
        pat = "|".join(re.escape(k) for k in keywords if k)
        kw_match = (
            plan_name_series.str.contains(pat, case=False, na=False, regex=True)
            if pat
            else pd.Series(False, index=plan_name_series.index)
        )
    else:
        kw_match = pd.Series(False, index=plan_name_series.index)
    return name_match | kw_match


# =========================================================
# Performance-focused compute
# =========================================================
def compute_removals_fast(input_long, db_small, selected_pairs, pair_types_map, active_plan_rules):
    if not selected_pairs:
        return {}

    if input_long is None or input_long.empty or db_small is None or db_small.empty:
        return {}

    tabs_accum = []

    selected_pairs_by_cls = {}
    for carrier, cls in selected_pairs:
        selected_pairs_by_cls.setdefault(cls, set()).add(carrier)

    selected_plan_ids = input_long["PlanId"].dropna().astype(str).unique()
    db_small_filtered = db_small[db_small["Plan_ID"].isin(selected_plan_ids)].copy()
    if db_small_filtered.empty:
        return {}

    for cls, carriers in selected_pairs_by_cls.items():
        db_cls = db_small_filtered[
            (db_small_filtered["Plan Classification"].astype(str).map(normalize_str) == normalize_str(cls)) &
            (db_small_filtered["Carrier_Name"].astype(str).map(normalize_str).isin(carriers))
        ].copy()

        if db_cls.empty:
            continue

        restricted_for_cls = {}
        no_match_pairs = set()

        for carrier in carriers:
            val = pair_types_map.get((carrier, cls), set())
            if val is None:
                no_match_pairs.add((carrier, cls))
            elif isinstance(val, set) and len(val) > 0:
                restricted_for_cls[(carrier, cls)] = val

        if no_match_pairs:
            no_match_carriers = set([c for c, _ in no_match_pairs])
            db_cls = db_cls[~db_cls["Carrier_Name"].isin(no_match_carriers)].copy()
            if db_cls.empty:
                continue

        if restricted_for_cls:
            keep_parts = []
            restricted_carriers = set([c for (c, _) in restricted_for_cls.keys()])
            unrestricted_carriers = carriers - restricted_carriers

            if unrestricted_carriers:
                keep_parts.append(db_cls[db_cls["Carrier_Name"].isin(unrestricted_carriers)].copy())

            for (carrier, _cls), allowed_types in restricted_for_cls.items():
                sub = db_cls[
                    (db_cls["Carrier_Name"] == carrier) &
                    (db_cls["Plan_Type"].astype(str).map(normalize_str).isin(allowed_types))
                ].copy()
                if not sub.empty:
                    keep_parts.append(sub)

            if keep_parts:
                db_cls = pd.concat(keep_parts, ignore_index=True)
            else:
                db_cls = db_cls.iloc[0:0].copy()

            if db_cls.empty:
                continue

        if active_plan_rules:
            keep_mask = pd.Series(True, index=db_cls.index)

            for carrier in carriers:
                rule = active_plan_rules.get((cls, carrier))
                if not rule:
                    continue

                mode = rule.get("mode", "ALL")
                if mode == "ALL":
                    continue

                names = set(rule.get("names", set()) or set())
                kws = list(rule.get("keywords", []) or [])

                sub_idx = db_cls.index[db_cls["Carrier_Name"] == carrier]
                if len(sub_idx) == 0:
                    continue

                sub = db_cls.loc[sub_idx]
                plan_names = sub["Plan_Name"].astype(str).map(normalize_str)
                m = _rule_match_series(plan_names, names, kws)

                if mode == "ONLY":
                    keep_mask.loc[sub_idx] = m.values
                elif mode == "ALL_EXCEPT":
                    keep_mask.loc[sub_idx] = (~m).values

            db_cls = db_cls[keep_mask].copy()
            if db_cls.empty:
                continue

        df_cls = input_long.merge(
            db_cls,
            left_on="PlanId",
            right_on="Plan_ID",
            how="inner",
            copy=False,
        )

        if df_cls.empty:
            continue

        out_df = df_cls[["MappingLevel", "ProviderId", "LocationId", "PlanId"]].drop_duplicates(
            subset=["MappingLevel", "ProviderId", "LocationId", "PlanId"]
        )

        if not out_df.empty:
            tabs_accum.append(out_df)

        del df_cls
        del db_cls
        gc.collect()

    if not tabs_accum:
        return {}

    final_df = pd.concat(tabs_accum, ignore_index=True).drop_duplicates(
        subset=["MappingLevel", "ProviderId", "LocationId", "PlanId"]
    )

    tabs = {}
    for lvl, g in final_df.groupby("MappingLevel", sort=False):
        tabs[str(lvl)] = g[["ProviderId", "LocationId", "PlanId"]].reset_index(drop=True)

    del final_df
    gc.collect()

    return tabs


def group_plans_comma(tabs):
    out = {}
    for lvl, df in tabs.items():
        if df is None or df.empty:
            out[lvl] = pd.DataFrame(columns=["ProviderId", "LocationId", "PlanIds"])
            continue

        g = (
            df.groupby(["ProviderId", "LocationId"], as_index=False)["PlanId"]
            .apply(lambda s: ",".join(sorted(set([str(x).strip() for x in s if str(x).strip()]))))
        )
        g = g.rename(columns={"PlanId": "PlanIds"})
        out[lvl] = g

    return out


def make_excel_bytes(tabs):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, df in tabs.items():
            safe_name = (sheet or "UnknownLevel")[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    bio.seek(0)
    return bio.getvalue()


def apply_default_types_to_selected_pairs(active_cls, default_types):
    if not active_cls:
        return

    default_types_set = set(default_types or [])
    new_map = dict(st.session_state.pair_types_map)

    for carrier, cls in st.session_state.selected_pairs:
        if active_cls != "(all classifications)" and cls != active_cls:
            continue

        if default_types_set:
            new_map[(carrier, cls)] = set(default_types_set)
        else:
            new_map[(carrier, cls)] = set()

    st.session_state.pair_types_map = new_map


def init_session_state():
    defaults = {
        "selected_pairs": set(),
        "pair_types_map": {},
        "active_plan_rules": {},
        "loaded": False,
        "active_global_class_filter": None,
        "prev_class_filter": None,
        "carrier_add_picker": [],
        "bulk_default_types": [],
        "explorer_selected_names": {},
        "explorer_carriers_selected": [],
        "debug_stats": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    init_session_state()

    st.title("Carrier Removal Tool")

    st.markdown(
        """
This tool helps you:
1. Upload your input and DB files  
2. Choose a **Plan Classification**  
3. Select the **Carriers** to process  
4. Choose **Plan Types** for all carriers or customize per carrier  
5. Optionally narrow down using **Plan Names**  
6. Preview and download the removal output
"""
    )

    st.info(
        "📌 **DB file requirements (.xlsx / .xls / .csv):**\n"
        f"- Must contain these columns exactly: **{', '.join(DB_REQUIRED_COLS)}**\n"
        "- **Plan_ID** values should include the `ip_` prefix (example: `ip_20356`).\n\n"
        "📌 **Input CSV requirements:**\n"
        f"- Must contain these columns: **{', '.join(INPUT_REQUIRED_COLS)}**"
    )

    # =========================================================
    # Uploads
    # =========================================================
    st.markdown("## Step 1: Upload files")

    col_up1, col_up2, col_up3 = st.columns([1, 1, 1])
    with col_up1:
        input_file = st.file_uploader(
            "Upload Input CSV",
            type=["csv"],
            help="Upload the input file that contains PracticeId, ProviderId, LocationId, PlanIds, and MappingLevel.",
        )
    with col_up2:
        db_file = st.file_uploader(
            "Upload DB file",
            type=["xlsx", "xls", "csv"],
            help="Upload the DB mapping file with carrier, plan, type, and classification details.",
        )
    with col_up3:
        dep_file = st.file_uploader(
            "Upload Deprecated Plans file (optional)",
            type=["xlsx", "xls", "csv"],
            help="Optional. Upload this if you want to compare missing Plan IDs against deprecated plans.",
        )

    load_btn = st.button("Load and analyze files", type="primary")

    if input_file is not None:
        st.write("Input file size (MB):", round(input_file.size / (1024 * 1024), 2))
    if db_file is not None:
        st.write("DB file size (MB):", round(db_file.size / (1024 * 1024), 2))

    # =========================================================
    # Load
    # =========================================================
    if load_btn:
        if input_file is None or db_file is None:
            st.error("Please upload both the Input CSV and the DB file.")
        else:
            prog = st.progress(0)
            status = st.empty()

            try:
                input_bytes = input_file.getvalue()
                db_bytes = db_file.getvalue()
                dep_bytes = dep_file.getvalue() if dep_file is not None else None

                prog.progress(10)
                status.info("Reading input file...")
                input_df = preprocess_input_df(input_bytes, input_file.name)

                prog.progress(30)
                status.info("Reading DB file...")
                db_df = preprocess_db_df(db_bytes, db_file.name)

                prog.progress(55)
                status.info("Analyzing carriers, classifications, and plan types...")
                universe = analyze_universe_from_frames(input_df, db_df)
                missing_plan_ids = universe["missing_plan_ids"]

                deprecated_found = 0
                deprecated_total = 0
                if dep_bytes is not None:
                    prog.progress(70)
                    status.info("Reading deprecated plans file...")
                    dep_ids = preprocess_deprecated_ids(dep_bytes, dep_file.name)
                    deprecated_total = len(dep_ids)
                    deprecated_found = len(missing_plan_ids.intersection(dep_ids))
                    del dep_ids
                    gc.collect()

                prog.progress(82)
                status.info("Preparing long input and DB slices...")
                input_long = build_input_long(input_df)
                db_small = db_df[DB_SMALL_COLS].copy()

                prog.progress(92)
                status.info("Saving prepared data...")
                st.session_state.input_long = input_long
                st.session_state.db_small = db_small
                st.session_state.universe = universe
                st.session_state.all_classifications = sorted(
                    list(universe["classification_to_carriers"].keys()),
                    key=lambda x: x.lower()
                )
                st.session_state.all_types = sorted(
                    list(universe["plan_types_universe"]),
                    key=lambda x: x.lower()
                )
                st.session_state.missing_plan_ids_count = len(missing_plan_ids)
                st.session_state.deprecated_found = deprecated_found
                st.session_state.deprecated_total = deprecated_total
                st.session_state.debug_stats = {
                    "input_rows": len(input_df),
                    "input_long_rows": len(input_long),
                    "db_rows_after_explode": len(db_df),
                    "db_small_rows": len(db_small),
                    "unique_input_plan_ids": len(extract_input_plan_ids(input_df)),
                    "missing_plan_ids": len(missing_plan_ids),
                }

                # Reset selections
                st.session_state.selected_pairs = set()
                st.session_state.pair_types_map = {}
                st.session_state.active_plan_rules = {}
                st.session_state.carrier_add_picker = []
                st.session_state.bulk_default_types = []
                st.session_state.explorer_selected_names = {}
                st.session_state.explorer_carriers_selected = []
                st.session_state.active_global_class_filter = None
                st.session_state.prev_class_filter = None

                st.session_state.loaded = True
                prog.progress(100)
                status.success("Files loaded successfully.")

                del input_df
                del db_df
                gc.collect()

            except Exception as e:
                prog.progress(100)
                status.error(f"Load failed: {e}")
                st.code(traceback.format_exc())

    # =========================================================
    # Main UI
    # =========================================================
    if st.session_state.loaded:
        universe = st.session_state.universe
        classification_to_carriers = universe["classification_to_carriers"]
        classification_to_types = universe["classification_to_types"]
        class_carrier_to_types = universe["class_carrier_to_types"]
        carrier_to_plan_names = universe["carrier_to_plan_names"]

        with st.expander("Diagnostics", expanded=False):
            stats = st.session_state.get("debug_stats", {})
            if stats:
                st.write(stats)

        with st.sidebar:
            st.header("Current working view")

            class_filter = st.selectbox(
                "Plan Classification",
                options=["(select)", "(all classifications)"] + st.session_state.all_classifications,
                index=0,
                format_func=lambda x: (
                    "Select a classification" if x == "(select)"
                    else "All classifications" if x == "(all classifications)"
                    else x
                ),
                key="sidebar_class_filter",
                help="Choose the classification you want to work on, or select All classifications.",
            )

            st.divider()

            view_search = st.text_input(
                "Search carriers or plan names",
                value="",
                placeholder="Example: aetna or rocky",
                key="sidebar_view_search",
                help="This only helps you find items on screen. It does not affect the final output unless you select them.",
            )

            st.divider()
            miss = st.session_state.missing_plan_ids_count
            dep_found = st.session_state.get("deprecated_found", 0)
            if st.session_state.get("deprecated_total", 0):
                st.warning(f"Missing in DB: {miss} | Deprecated matched: {dep_found}")
            else:
                st.info(f"Missing in DB: {miss} (upload deprecated file to compare)")

        active_cls = None if class_filter == "(select)" else class_filter
        all_classifications_mode = active_cls == "(all classifications)"
        st.session_state.active_global_class_filter = active_cls
        disabled_pick = active_cls is None

        if class_filter != st.session_state.prev_class_filter:
            st.session_state.carrier_add_picker = []
            st.session_state.explorer_carriers_selected = []
            st.session_state.bulk_default_types = []
            st.session_state.prev_class_filter = class_filter

        if not active_cls:
            carriers_base = []
        elif all_classifications_mode:
            all_carriers = set()
            for carriers in classification_to_carriers.values():
                all_carriers.update(carriers)
            carriers_base = sorted(list(all_carriers), key=lambda x: x.lower())
        else:
            carriers_base = sorted(
                list(classification_to_carriers.get(active_cls, set())),
                key=lambda x: x.lower()
            )

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

        st.markdown("## Step 2 onward: Make your selections")

        left, right = st.columns([2.2, 1.2], gap="large")

        with left:
            st.markdown("### Step 2: Choose a plan classification")
            if disabled_pick:
                st.info("Start by selecting a Plan Classification from the left sidebar.")
            else:
                if all_classifications_mode:
                    st.success("Working on classification: **All classifications**")
                    st.caption(f"Carriers visible across all classifications: {len(displayed_carriers)}")
                else:
                    st.success(f"Working on classification: **{active_cls}**")
                    st.caption(f"Carriers visible in this classification: {len(displayed_carriers)}")

            st.markdown("### Step 3: Select carriers")

            def add_all_shown():
                cls = st.session_state.active_global_class_filter
                if not cls:
                    return

                for carrier in st.session_state.get("displayed_carriers_cache", []):
                    if cls == "(all classifications)":
                        for actual_cls in st.session_state.all_classifications:
                            if carrier in classification_to_carriers.get(actual_cls, set()):
                                pair = (carrier, actual_cls)
                                st.session_state.selected_pairs.add(pair)
                                st.session_state.pair_types_map.setdefault(pair, set())
                    else:
                        pair = (carrier, cls)
                        st.session_state.selected_pairs.add(pair)
                        st.session_state.pair_types_map.setdefault(pair, set())

            def add_selected_carriers():
                cls = st.session_state.active_global_class_filter
                if not cls:
                    return

                picked = st.session_state.get("carrier_add_picker", [])
                for carrier in picked:
                    if cls == "(all classifications)":
                        for actual_cls in st.session_state.all_classifications:
                            if carrier in classification_to_carriers.get(actual_cls, set()):
                                pair = (carrier, actual_cls)
                                st.session_state.selected_pairs.add(pair)
                                st.session_state.pair_types_map.setdefault(pair, set())
                    else:
                        pair = (carrier, cls)
                        st.session_state.selected_pairs.add(pair)
                        st.session_state.pair_types_map.setdefault(pair, set())

                st.session_state["carrier_add_picker"] = []

            def clear_all_selections():
                st.session_state.selected_pairs = set()
                st.session_state.pair_types_map = {}
                st.session_state.active_plan_rules = {}
                st.session_state.explorer_selected_names = {}
                st.session_state.explorer_carriers_selected = []
                st.session_state.carrier_add_picker = []
                st.session_state.bulk_default_types = []

            st.session_state["displayed_carriers_cache"] = list(displayed_carriers)

            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                st.button("Add all shown", disabled=disabled_pick, on_click=add_all_shown)
            with colB:
                st.button("Clear everything", on_click=clear_all_selections)
            with colC:
                selected_total = len(st.session_state.selected_pairs)
                selected_in_cls = (
                    len(st.session_state.selected_pairs)
                    if all_classifications_mode
                    else len([1 for (_, cls) in st.session_state.selected_pairs if cls == active_cls]) if active_cls else 0
                )
                st.caption(
                    f"Visible: {len(displayed_carriers)} | Selected in this classification: {selected_in_cls} | Total selected: {selected_total}"
                )

            st.multiselect(
                "Choose carriers to add",
                options=displayed_carriers,
                key="carrier_add_picker",
                disabled=disabled_pick,
                help="Pick one or more carriers, then click 'Add selected carriers'.",
            )
            st.button(
                "Add selected carriers",
                use_container_width=True,
                disabled=disabled_pick,
                on_click=add_selected_carriers,
            )

            st.divider()
            st.markdown("### Step 4: Choose plan types")
            st.caption(
                "You can apply one default plan type selection to all selected carriers in this classification, "
                "and then override it for specific carriers if needed."
            )

            selected_pairs_in_active_cls = sorted(
                [
                    (carrier, cls)
                    for (carrier, cls) in st.session_state.selected_pairs
                    if all_classifications_mode or cls == active_cls
                ],
                key=lambda x: (x[0].lower(), x[1].lower())
            )

            if not active_cls:
                available_types_for_class = []
            elif all_classifications_mode:
                all_types_for_view = set()
                for types_set in classification_to_types.values():
                    all_types_for_view.update(types_set)
                available_types_for_class = sorted(list(all_types_for_view), key=lambda x: x.lower())
            else:
                available_types_for_class = sorted(
                    list(classification_to_types.get(active_cls, set())),
                    key=lambda x: x.lower()
                )

            if disabled_pick:
                st.info("Select a classification first.")
            elif not selected_pairs_in_active_cls:
                st.info("Select at least one carrier before choosing plan types.")
            else:
                st.markdown("#### Default plan types for all selected carriers")
                st.multiselect(
                    "Default plan types",
                    options=available_types_for_class,
                    key="bulk_default_types",
                    help=(
                        "Leave this empty to include all plan types. "
                        "If you choose one or more values, those plan types will be used for all selected carriers in this classification."
                    ),
                )

                def apply_default_plan_types():
                    apply_default_types_to_selected_pairs(active_cls, st.session_state.bulk_default_types)

                st.button(
                    "Apply default plan types to selected carriers",
                    use_container_width=True,
                    on_click=apply_default_plan_types,
                )

                st.markdown("#### Optional carrier-level overrides")
                st.caption("Use this only when one carrier needs different plan types from the default.")

                MAX_CARRIER_TYPE_RENDER = 40
                if len(selected_pairs_in_active_cls) > MAX_CARRIER_TYPE_RENDER:
                    st.warning(
                        f"Showing the first {MAX_CARRIER_TYPE_RENDER} selected carriers in this section. "
                        "If needed, reduce the number of selected carriers."
                    )

                for carrier, cls in selected_pairs_in_active_cls[:MAX_CARRIER_TYPE_RENDER]:
                    carrier_types = sorted(
                        list(class_carrier_to_types.get((cls, carrier), set())),
                        key=lambda x: x.lower()
                    )

                    mode_key = type_override_mode_key_for(cls, carrier)
                    values_key = type_override_values_key_for(cls, carrier)

                    current_val = st.session_state.pair_types_map.get((carrier, cls), set())

                    if mode_key not in st.session_state:
                        st.session_state[mode_key] = "Use default / all"

                    if values_key not in st.session_state:
                        if isinstance(current_val, set) and current_val:
                            st.session_state[values_key] = sorted(list(current_val), key=lambda x: x.lower())
                        else:
                            st.session_state[values_key] = []

                    with st.expander(f"{carrier} ({cls})", expanded=False):
                        st.radio(
                            "How should this carrier use plan types?",
                            options=[
                                "Use default / all",
                                "Only selected plan types",
                            ],
                            key=mode_key,
                        )

                        st.multiselect(
                            f"Plan types for {carrier}",
                            options=carrier_types,
                            key=values_key,
                        )

                        def save_type_override(c=carrier, cl=cls, mk=mode_key, vk=values_key):
                            mode = st.session_state.get(mk, "Use default / all")
                            selected_vals = st.session_state.get(vk, [])

                            if mode == "Only selected plan types":
                                st.session_state.pair_types_map[(c, cl)] = set(selected_vals)
                            else:
                                if st.session_state.get("bulk_default_types", []):
                                    st.session_state.pair_types_map[(c, cl)] = set(st.session_state["bulk_default_types"])
                                else:
                                    st.session_state.pair_types_map[(c, cl)] = set()

                        st.button(
                            f"Save plan type setting for {carrier}",
                            key=f"save_type_override_{stable_key(cls, carrier)}",
                            on_click=save_type_override,
                            use_container_width=True,
                        )

            st.divider()
            st.markdown("### Step 5: Optional plan name filtering")
            st.caption("Use this section only if you want to narrow the removal list further by plan name.")

            explorer_kw = st.text_input(
                "Search plan names by keyword",
                value="",
                placeholder="Example: rocky or denver",
                disabled=disabled_pick,
                key="explorer_kw",
            )

            explorer_mode = st.radio(
                "How should selected plan names be treated?",
                options=[
                    "Keep only these plan names",
                    "Remove only these plan names",
                ],
                index=0,
                disabled=disabled_pick,
                key="explorer_mode",
            )

            by_carrier = {}
            if (not disabled_pick) and explorer_kw.strip():
                kw = explorer_kw.strip()
                dbs = st.session_state.db_small.copy()
                dbs["Carrier_Name"] = dbs["Carrier_Name"].astype(str).map(normalize_str)
                dbs["Plan Classification"] = dbs["Plan Classification"].astype(str).map(normalize_str)
                dbs["Plan_Name"] = dbs["Plan_Name"].astype(str).map(normalize_str)

                if all_classifications_mode:
                    sub = dbs
                else:
                    sub = dbs[dbs["Plan Classification"] == active_cls]

                sub = sub[sub["Plan_Name"].str.contains(re.escape(kw), case=False, na=False)]

                if not sub.empty:
                    sub = sub[["Carrier_Name", "Plan_Name"]].drop_duplicates()
                    for car, pname in sub.itertuples(index=False, name=None):
                        by_carrier.setdefault(car, set()).add(pname)

            carriers_found = sorted(list(by_carrier.keys()), key=lambda x: x.lower())

            if disabled_pick:
                st.info("Select a classification first to use plan name filtering.")
            elif not explorer_kw.strip():
                st.info("Enter a keyword if you want to filter by plan name.")
            elif not carriers_found:
                st.warning("No matching plan names found for this classification.")
            else:
                total_plans = sum(len(v) for v in by_carrier.values())
                st.success(f"Found {total_plans} matching plan names across {len(carriers_found)} carriers.")

                if not st.session_state.explorer_carriers_selected:
                    st.session_state.explorer_carriers_selected = list(carriers_found)

                def explorer_select_all_carriers():
                    st.session_state["explorer_carriers_selected"] = list(carriers_found)

                def explorer_clear_all_carriers():
                    st.session_state["explorer_carriers_selected"] = []

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.button("Select all carriers found", use_container_width=True, on_click=explorer_select_all_carriers)
                with cc2:
                    st.button("Clear carrier selection", use_container_width=True, on_click=explorer_clear_all_carriers)

                st.multiselect(
                    "Choose carriers for this plan-name rule",
                    options=carriers_found,
                    key="explorer_carriers_selected",
                )
                selected_carriers_for_rule = list(st.session_state.explorer_carriers_selected)

                def explorer_select_all_plans_global():
                    cls = st.session_state.active_global_class_filter
                    if not cls:
                        return
                    for car in selected_carriers_for_rule:
                        plans = sorted(list(by_carrier.get(car, set())), key=lambda x: (x or "").lower())
                        st.session_state.explorer_selected_names[(cls, car)] = set(plans)
                        st.session_state[ms_key_for(cls, car)] = plans

                def explorer_clear_all_plans_global():
                    cls = st.session_state.active_global_class_filter
                    if not cls:
                        return
                    for car in selected_carriers_for_rule:
                        st.session_state.explorer_selected_names[(cls, car)] = set()
                        st.session_state[ms_key_for(cls, car)] = []

                pp1, pp2 = st.columns(2)
                with pp1:
                    st.button("Select all plan names shown", use_container_width=True, on_click=explorer_select_all_plans_global)
                with pp2:
                    st.button("Clear all selected plan names", use_container_width=True, on_click=explorer_clear_all_plans_global)

                MAX_RENDER = 25
                if len(selected_carriers_for_rule) > MAX_RENDER:
                    st.warning(f"Showing the first {MAX_RENDER} carriers. Narrow the keyword if needed.")
                render_carriers = selected_carriers_for_rule[:MAX_RENDER]

                for car in render_carriers:
                    plans = sorted(list(by_carrier.get(car, set())), key=lambda x: (x or "").lower())
                    bucket = (active_cls, car)

                    msk = ms_key_for(active_cls, car)
                    if msk not in st.session_state:
                        current = set(st.session_state.explorer_selected_names.get(bucket, set()) or set())
                        st.session_state[msk] = sorted(list(current.intersection(set(plans))), key=lambda x: (x or "").lower())

                    pk = proc_key_for(active_cls, car)
                    if pk not in st.session_state:
                        if all_classifications_mode:
                            st.session_state[pk] = any(
                                (car, cls_name) in st.session_state.selected_pairs
                                for cls_name in st.session_state.all_classifications
                            )
                        else:
                            st.session_state[pk] = ((car, active_cls) in st.session_state.selected_pairs)

                    selected_count = len(set(st.session_state[msk]))
                    exp_label = f"{car} — selected {selected_count} of {len(plans)}"

                    with st.expander(exp_label, expanded=False):
                        st.checkbox("Apply this plan-name rule to this carrier", key=pk)

                        def _sel_all_carrier(carrier=car):
                            cls = st.session_state.active_global_class_filter
                            if not cls:
                                return
                            allp = sorted(list(by_carrier.get(carrier, set())), key=lambda x: (x or "").lower())
                            st.session_state.explorer_selected_names[(cls, carrier)] = set(allp)
                            st.session_state[ms_key_for(cls, carrier)] = allp

                        def _clear_carrier(carrier=car):
                            cls = st.session_state.active_global_class_filter
                            if not cls:
                                return
                            st.session_state.explorer_selected_names[(cls, carrier)] = set()
                            st.session_state[ms_key_for(cls, carrier)] = []

                        b1, b2 = st.columns(2)
                        with b1:
                            st.button(
                                "Select all for this carrier",
                                use_container_width=True,
                                on_click=_sel_all_carrier,
                                key=f"sel_all_{stable_key(active_cls, car)}",
                            )
                        with b2:
                            st.button(
                                "Clear for this carrier",
                                use_container_width=True,
                                on_click=_clear_carrier,
                                key=f"clr_{stable_key(active_cls, car)}",
                            )

                        picked = st.multiselect("Plan names", options=plans, key=msk)
                        st.session_state.explorer_selected_names[bucket] = set(picked)

                st.divider()

                def apply_explorer_rule():
                    cls = st.session_state.active_global_class_filter
                    if not cls:
                        return

                    mode_label = st.session_state.get("explorer_mode", "")
                    rule_mode = "ALL_EXCEPT" if mode_label.startswith("Keep only") else "ONLY"

                    applied = 0
                    added_to_summary = 0

                    for car in selected_carriers_for_rule:
                        if not st.session_state.get(proc_key_for(cls, car), False):
                            continue

                        names = set(st.session_state.explorer_selected_names.get((cls, car), set()) or set())
                        if not names:
                            continue

                        if all_classifications_mode:
                            matching_classes = [
                                actual_cls
                                for actual_cls in st.session_state.all_classifications
                                if car in classification_to_carriers.get(actual_cls, set())
                            ]
                        else:
                            matching_classes = [cls]

                        for actual_cls in matching_classes:
                            pair = (car, actual_cls)
                            if pair not in st.session_state.selected_pairs:
                                st.session_state.selected_pairs.add(pair)
                                if st.session_state.get("bulk_default_types", []):
                                    st.session_state.pair_types_map[pair] = set(st.session_state["bulk_default_types"])
                                else:
                                    st.session_state.pair_types_map.setdefault(pair, set())
                                added_to_summary += 1

                            st.session_state.active_plan_rules[(actual_cls, car)] = {
                                "mode": rule_mode,
                                "names": set(names),
                                "keywords": [],
                            }
                            applied += 1

                    if applied == 0:
                        st.warning("Nothing was applied. Select at least one plan name and choose at least one carrier.")
                    else:
                        msg = f"Applied plan-name rules to {applied} carrier/classification combinations."
                        if added_to_summary:
                            msg += f" Added {added_to_summary} carriers into your selection."
                        st.success(msg)

                def clear_rules_for_this_class():
                    cls = st.session_state.active_global_class_filter
                    if not cls:
                        return

                    if all_classifications_mode:
                        st.session_state.active_plan_rules = {}
                        st.success("Cleared active plan-name rules for all classifications.")
                    else:
                        to_del = [k for k in list(st.session_state.active_plan_rules.keys()) if k[0] == cls]
                        for k in to_del:
                            st.session_state.active_plan_rules.pop(k, None)
                        st.success(f"Cleared active plan-name rules for {cls}.")

                ap1, ap2 = st.columns(2)
                with ap1:
                    st.button("Confirm and apply plan-name rules", use_container_width=True, on_click=apply_explorer_rule)
                with ap2:
                    st.button("Clear active rules for this classification", use_container_width=True, on_click=clear_rules_for_this_class)

        with right:
            st.markdown("### Step 6: Review your selection")

            selected_pairs_sorted = sorted(
                list(st.session_state.selected_pairs),
                key=lambda x: (x[0].lower(), x[1].lower())
            )

            if not selected_pairs_sorted:
                st.info("Select carriers to enable preview and output.")
            else:
                rows = []
                for (carrier, cls) in selected_pairs_sorted:
                    types_val = st.session_state.pair_types_map.get((carrier, cls), set())
                    rule = st.session_state.active_plan_rules.get((cls, carrier))
                    if rule is None:
                        rule_mode = "No plan-name rule"
                        rule_count = 0
                    else:
                        rule_mode = "Keep only selected names" if rule.get("mode") == "ALL_EXCEPT" else "Remove only selected names"
                        rule_count = len(rule.get("names", set()) or set())

                    rows.append(
                        {
                            "Carrier": carrier,
                            "Plan Classification": cls,
                            "Plan Types": get_pair_type_summary(types_val),
                            "Plan Name Rule": rule_mode,
                            "Selected Plan Names": rule_count,
                        }
                    )

                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                st.divider()
                st.markdown("### Step 7: Preview and download")

                if st.button("Preview removal counts", use_container_width=True):
                    try:
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
                        st.dataframe(preview_df, use_container_width=True)

                        del tabs_preview
                        del preview_df
                        gc.collect()

                    except Exception as e:
                        st.error(f"Preview failed: {e}")
                        st.code(traceback.format_exc())

                st.divider()

                output_format = st.radio(
                    "Choose output format",
                    options=[
                        "Current output (one row per PlanId)",
                        "Grouped output (comma-separated PlanIds per ProviderId + LocationId)",
                    ],
                    index=0,
                    key="output_format_choice",
                )

                generate = st.button("Generate final removal output", type="primary", use_container_width=True)
                if generate:
                    try:
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
                            st.warning("No removals matched your current selection.")
                        else:
                            if output_format.startswith("Grouped"):
                                tabs_to_write = group_plans_comma(tabs_final)
                                file_name = "removal_output_grouped.xlsx"
                            else:
                                tabs_to_write = tabs_final
                                file_name = "removal_output.xlsx"

                            xbytes = make_excel_bytes(tabs_to_write)

                            st.success(f"Removal output is ready. Tabs created: {len(tabs_to_write)}")
                            st.download_button(
                                f"Download {file_name}",
                                data=xbytes,
                                file_name=file_name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )

                        del tabs_final
                        gc.collect()

                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.code(traceback.format_exc())


try:
    main()
except Exception as e:
    st.error("The app hit an unexpected error.")
    st.error(str(e))
    st.code(traceback.format_exc())
