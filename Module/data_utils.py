"""
data_utils.py
=============
Data loading, stratified sampling, preprocessing, and train/test splitting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import config

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'ARP', 'BOOTP_options', 'BOOTP_secs', 'BOOTP_xid', 'Burst_count',
    'DHCP_options', 'DNS_Interval', 'DNS_Query_Len', 'DNS_Query_Type',
    'DNS_aa', 'DNS_arcount', 'DNS_cd', 'DNS_id', 'DNS_qdcount', 'DNS_query',
    'DNS_rd', 'DNS_tc', 'Direction_ratio', 'Entropy', 'Entropy_Q1',
    'Entropy_Var', 'Ether_type', 'HTTPS', 'HTTP_Content_Len', 'HTTP_URI',
    'HTTP_content_type', 'HTTP_host', 'HTTP_status', 'IAT_mean', 'IAT_std',
    'ICMP_chksum', 'ICMP_length', 'ICMP_type', 'IP_MF', 'IP_flags',
    'IP_frag', 'IP_id', 'IP_len', 'IP_options', 'IP_padding', 'IP_proto',
    'IP_tos', 'LLC_dsap', 'LLC_ssap', 'Min_Elapsed_Time', 'NTP',
    'NTP_Interval', 'Packet_freq', 'Pck_Size_Avg', 'Pck_Size_IQR',
    'Pck_Size_Max', 'Pck_Size_Med', 'Pck_Size_Min', 'Pck_Size_Q1',
    'Pck_Size_Q3', 'Pck_Size_Sum', 'Pck_Size_Var', 'Protocol_ICMP_Ratio',
    'Protocol_TCP_Ratio', 'TCP_PSH', 'TCP_URG', 'TCP_dport', 'TCP_flags',
    'TCP_mss_values', 'TCP_options', 'TCP_reserved', 'TCP_response',
    'TCP_seq', 'TCP_sport', 'TCP_window_scaling', 'TLS_ja3',
    'TLS_selected_cipher', 'TLS_version', 'Time_Since_Prev_Frame',
    'UDP_chksum', 'UDP_dport', 'UDP_len', 'UDP_sport', 'dport_bare',
    'sport23',
]
LABEL_COLS = ['Label', 'Traffic Type']
ALL_COLS   = FEATURE_COLS + LABEL_COLS


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def sample_by_labels(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    Stratified proportional sampling per class (or joint class pair).

    Parameters
    ----------
    df            : input DataFrame
    target_column : 'Label', 'Traffic Type', or None (→ joint sampling)
    """
    frac = config.SAMPLE_FRAC_SINGLE if target_column else config.SAMPLE_FRAC_MULTI

    if target_column:
        if target_column not in df.columns:
            print(f"[WARN] '{target_column}' not found — returning df unchanged.")
            return df
        print(f"\nClass counts before sampling ({target_column}):")
        print(df[target_column].value_counts())
        sampled = (df
                   .groupby(target_column, group_keys=False)
                   .apply(lambda g: g.sample(frac=frac, random_state=config.RANDOM_SEED))
                   .reset_index(drop=True))
        print(f"\nClass counts after sampling ({frac*100:.0f}%):")
        print(sampled[target_column].value_counts())
    else:
        if not {'Label', 'Traffic Type'}.issubset(df.columns):
            print("[WARN] 'Label'/'Traffic Type' missing — returning df unchanged.")
            return df
        print("\n(Label, Traffic Type) counts before sampling:")
        print(df.groupby(['Label', 'Traffic Type']).size())
        sampled = (df
                   .groupby(['Label', 'Traffic Type'], group_keys=False)
                   .apply(lambda g: g.sample(frac=frac, random_state=config.RANDOM_SEED))
                   .reset_index(drop=True))
        print(f"\n(Label, Traffic Type) counts after sampling ({frac*100:.0f}%):")
        print(sampled.groupby(['Label', 'Traffic Type']).size())

    print(f"\nOriginal: {len(df):,} rows → Sampled: {len(sampled):,} rows")
    return sampled


def load_and_preprocess(
    csv_path: Path,
    target_labels=None,
    target_traffic_types=None,
    output_mode: str = 'multi',
):
    """
    Load CSV, filter, deduplicate, sample, encode, and scale.

    Returns
    -------
    X_scaled      : np.ndarray  (n_samples, n_features)
    y_encoded     : np.ndarray or tuple(np.ndarray, np.ndarray)
    le            : LabelEncoder or tuple(LabelEncoder, LabelEncoder)
    feature_names : Index of feature column names
    """
    # -- Load with column subset -------------------------------------------
    available = pd.read_csv(csv_path, nrows=1).columns.tolist()
    usecols   = [c for c in ALL_COLS if c in available]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    print(f"\nLoaded {csv_path.name}: {len(df):,} rows × {len(df.columns)} cols")

    # -- Optional filters ---------------------------------------------------
    if target_labels is not None and 'Label' in df.columns:
        missing = [l for l in target_labels if l not in df['Label'].values]
        if missing:
            print(f"[WARN] Labels not found: {missing}")
        df = df[df['Label'].isin(target_labels)]
        print(f"After label filter: {len(df):,} rows")

    if target_traffic_types is not None and 'Traffic Type' in df.columns:
        missing = [t for t in target_traffic_types if t not in df['Traffic Type'].values]
        if missing:
            print(f"[WARN] Traffic types not found: {missing}")
        df = df[df['Traffic Type'].isin(target_traffic_types)]
        print(f"After traffic-type filter: {len(df):,} rows")

    if df.empty:
        raise ValueError("No data remains after filtering.")

    # -- Basic QA ----------------------------------------------------------
    print(f"\nNull counts:\n{df.isna().sum()[df.isna().sum() > 0]}")
    n_dup = df.duplicated().sum()
    if n_dup:
        df.drop_duplicates(inplace=True)
        print(f"Dropped {n_dup:,} duplicate rows → {len(df):,} remain")

    # Inject synthetic 'Traffic Type' if missing and needed
    if output_mode == 'traffic' and 'Traffic Type' not in df.columns:
        print("[WARN] 'Traffic Type' missing — synthesising from 'Label'.")
        df['Traffic Type'] = df['Label'].apply(
            lambda x: 'Attack' if 'attack' in str(x).lower() else 'Benign'
        )

    # -- Sampling ----------------------------------------------------------
    target_col = {'device': 'Label', 'traffic': 'Traffic Type'}.get(output_mode)
    df = sample_by_labels(df, target_col)

    # -- Validate required columns -----------------------------------------
    if output_mode in ('device', 'multi') and 'Label' not in df.columns:
        raise ValueError("'Label' column required but not found.")
    if output_mode in ('traffic', 'multi') and 'Traffic Type' not in df.columns:
        raise ValueError("'Traffic Type' column required but not found.")

    # -- Drop rare classes (support == 1) ----------------------------------
    df = _drop_rare_classes(df, output_mode)
    print(f"\nAfter dropping rare classes: {len(df):,} rows")

    # -- Split X / y -------------------------------------------------------
    drop_cols = ['Label', 'Traffic Type'] if output_mode == 'multi' else [target_col]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    # -- Encode non-numeric features ---------------------------------------
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le_tmp = LabelEncoder()
        X[col] = le_tmp.fit_transform(X[col].astype(str))

    if not all(X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))):
        raise ValueError("Non-numeric columns remain after encoding.")

    # -- Scale -------------------------------------------------------------
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.array(X_scaled, dtype=np.float32)

    # -- Encode targets ----------------------------------------------------
    if output_mode == 'multi':
        le_device = LabelEncoder()
        le_attack = LabelEncoder()
        y_device  = le_device.fit_transform(df['Label'])
        y_attack  = le_attack.fit_transform(df['Traffic Type'])
        y_encoded = (y_device, y_attack)
        le        = (le_device, le_attack)
    else:
        le        = LabelEncoder()
        y_encoded = le.fit_transform(df[target_col])

    print(f"\nFeature matrix shape : {X_scaled.shape}")
    if output_mode == 'multi':
        print(f"Device labels shape  : {y_encoded[0].shape}")
        print(f"Attack labels shape  : {y_encoded[1].shape}")
    else:
        print(f"Target labels shape  : {y_encoded.shape}")

    return X_scaled, y_encoded, le, X.columns


def split_data(X, y_encoded, output_mode: str):
    """
    Stratified train/test split.

    Returns a dict with keys:
      X_train, X_test,
      y_train, y_test          (single) or
      y_device_train, y_device_test,
      y_attack_train, y_attack_test,
      y_train, y_test           (stacked, for multi-output sklearn models)
    """
    rs = config.RANDOM_SEED
    ts = config.TEST_SIZE

    if output_mode == 'multi':
        y_device, y_attack = y_encoded
        strat = np.column_stack((y_device, y_attack))
        (X_train, X_test,
         yd_train, yd_test,
         ya_train, ya_test) = train_test_split(
            X, y_device, y_attack,
            test_size=ts, random_state=rs,
            stratify=strat,
        )
        y_train = np.column_stack((ya_train, yd_train))
        y_test  = np.column_stack((ya_test,  yd_test))
        return dict(
            X_train=X_train, X_test=X_test,
            y_device_train=yd_train, y_device_test=yd_test,
            y_attack_train=ya_train, y_attack_test=ya_test,
            y_train=y_train, y_test=y_test,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=ts, random_state=rs,
            stratify=y_encoded,
        )
        return dict(X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _drop_rare_classes(df: pd.DataFrame, output_mode: str) -> pd.DataFrame:
    if output_mode == 'multi':
        for col in ('Label', 'Traffic Type'):
            counts = df[col].value_counts()
            rare   = counts[counts == 1].index
            if len(rare):
                print(f"Dropping rare {col} classes (n=1): {rare.tolist()}")
            df = df[~df[col].isin(rare)]
        combo = df.groupby(['Label', 'Traffic Type']).size()
        rare_combo = combo[combo == 1].index
        if len(rare_combo):
            print(f"Dropping rare (Label, Traffic Type) combos: {rare_combo.tolist()}")
        df = df[~df[['Label', 'Traffic Type']].apply(tuple, axis=1).isin(rare_combo)]
    else:
        col    = 'Label' if output_mode == 'device' else 'Traffic Type'
        counts = df[col].value_counts()
        rare   = counts[counts == 1].index
        if len(rare):
            print(f"Dropping rare {col} classes (n=1): {rare.tolist()}")
        df = df[~df[col].isin(rare)]
    return df
