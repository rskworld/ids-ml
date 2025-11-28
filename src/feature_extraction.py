"""
Feature extraction module for network traffic data.
Extracts features from network packets and flows.
"""

import numpy as np
import pandas as pd
from collections import Counter


class NetworkFeatureExtractor:
    """Extract features from network traffic data."""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_basic_features(self, df):
        """
        Extract basic statistical features from network data.
        
        Args:
            df: DataFrame with network traffic data
            
        Returns:
            DataFrame: DataFrame with extracted features
        """
        features = {}
        
        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_min'] = df[col].min()
            features[f'{col}_max'] = df[col].max()
            features[f'{col}_median'] = df[col].median()
        
        # Count features
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            features[f'{col}_unique_count'] = df[col].nunique()
            features[f'{col}_mode'] = df[col].mode()[0] if len(df[col].mode()) > 0 else None
        
        return pd.DataFrame([features])
    
    def extract_flow_features(self, df, src_ip_col='src_ip', dst_ip_col='dst_ip',
                             src_port_col='src_port', dst_port_col='dst_port'):
        """
        Extract flow-based features.
        
        Args:
            df: DataFrame with network data
            src_ip_col: Source IP column name
            dst_ip_col: Destination IP column name
            src_port_col: Source port column name
            dst_port_col: Destination port column name
            
        Returns:
            DataFrame: DataFrame with flow features
        """
        features = {}
        
        # Check if required columns exist
        required_cols = [src_ip_col, dst_ip_col, src_port_col, dst_port_col]
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        # Flow statistics
        if src_ip_col in df.columns:
            features['unique_src_ips'] = df[src_ip_col].nunique()
            features['src_ip_entropy'] = self._calculate_entropy(df[src_ip_col])
        
        if dst_ip_col in df.columns:
            features['unique_dst_ips'] = df[dst_ip_col].nunique()
            features['dst_ip_entropy'] = self._calculate_entropy(df[dst_ip_col])
        
        if src_port_col in df.columns:
            features['unique_src_ports'] = df[src_port_col].nunique()
            features['src_port_entropy'] = self._calculate_entropy(df[src_port_col])
        
        if dst_port_col in df.columns:
            features['unique_dst_ports'] = df[dst_port_col].nunique()
            features['dst_port_entropy'] = self._calculate_entropy(df[dst_port_col])
        
        # Connection patterns
        if all(col in df.columns for col in [src_ip_col, dst_ip_col]):
            df['connection'] = df[src_ip_col].astype(str) + '->' + df[dst_ip_col].astype(str)
            features['unique_connections'] = df['connection'].nunique()
            features['connection_entropy'] = self._calculate_entropy(df['connection'])
        
        return pd.DataFrame([features])
    
    def extract_temporal_features(self, df, time_col='timestamp'):
        """
        Extract temporal features from network data.
        
        Args:
            df: DataFrame with network data
            time_col: Timestamp column name
            
        Returns:
            DataFrame: DataFrame with temporal features
        """
        features = {}
        
        if time_col not in df.columns:
            return pd.DataFrame()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Temporal statistics
        features['time_span_seconds'] = (df[time_col].max() - df[time_col].min()).total_seconds()
        features['packets_per_second'] = len(df) / max(features['time_span_seconds'], 1)
        
        # Hour of day features
        df['hour'] = df[time_col].dt.hour
        features['peak_hour'] = df['hour'].mode()[0] if len(df['hour'].mode()) > 0 else 0
        features['hour_entropy'] = self._calculate_entropy(df['hour'])
        
        return pd.DataFrame([features])
    
    def extract_protocol_features(self, df, protocol_col='protocol'):
        """
        Extract protocol-related features.
        
        Args:
            df: DataFrame with network data
            protocol_col: Protocol column name
            
        Returns:
            DataFrame: DataFrame with protocol features
        """
        features = {}
        
        if protocol_col not in df.columns:
            return pd.DataFrame()
        
        protocol_counts = df[protocol_col].value_counts()
        features['unique_protocols'] = len(protocol_counts)
        features['protocol_entropy'] = self._calculate_entropy(df[protocol_col])
        features['dominant_protocol'] = protocol_counts.index[0] if len(protocol_counts) > 0 else None
        features['dominant_protocol_ratio'] = protocol_counts.iloc[0] / len(df) if len(protocol_counts) > 0 else 0
        
        return pd.DataFrame([features])
    
    def _calculate_entropy(self, series):
        """
        Calculate Shannon entropy of a series.
        
        Args:
            series: Pandas Series
            
        Returns:
            float: Entropy value
        """
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def extract_all_features(self, df):
        """
        Extract all available features from network data.
        
        Args:
            df: DataFrame with network data
            
        Returns:
            DataFrame: DataFrame with all extracted features
        """
        all_features = []
        
        # Basic features
        basic_features = self.extract_basic_features(df)
        if not basic_features.empty:
            all_features.append(basic_features)
        
        # Flow features
        flow_features = self.extract_flow_features(df)
        if not flow_features.empty:
            all_features.append(flow_features)
        
        # Temporal features
        temporal_features = self.extract_temporal_features(df)
        if not temporal_features.empty:
            all_features.append(temporal_features)
        
        # Protocol features
        protocol_features = self.extract_protocol_features(df)
        if not protocol_features.empty:
            all_features.append(protocol_features)
        
        # Combine all features
        if all_features:
            result = pd.concat(all_features, axis=1)
            return result
        else:
            return pd.DataFrame()

