import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer

class OutlierRemover:
    """
    A flexible Outlier Removal and Impact Reduction utility.

     Supported Methods:
    - zscore
    - zscore_capper
    - yeo_johnson
    - yeo_johnson_capper
    - impact_reduction
    - adaptive_trimming
    - smooth_capping
    - local_standardization
    """
    def __init__(self,method="zscore",threshold=3.0,smooth_factor=0.9):
        """
        Args:
            method (str): Method to remove or reduce outliers.
            threshold (float): Threshold for outlier detection.
        """
        self.method=method
        self.threshold=threshold
        self.smooth_factor=smooth_factor
        
        
    def fit_transform(self,data):
        """
        Apply selected outlier removal technique.

        Args:
            data (pd.DataFrame or pd.Series): Input data.

        Returns:
            pd.DataFrame: Processed data after outlier handling.
        """
        methods={
            "zscore": self._zscore_removal,
            "zscore_capper": self._zscore_capper,
            "yeo_johnson": self._yeo_johnson_removal,
            "yeo_johnson_capper": self._yeo_johnson_capper,
            "impact_reduction": self._reduce_impact,
            "adaptive_trimming": self._adaptive_trimming,
            "smooth_capping": self._smooth_boundary_capping,
            "local_standardization": self._local_standardization_correction,
        }
        if self.method not in methods:
           raise ValueError(f"Unknown method {self.method}")
    
        return methods[self.method](data)
    def multi_outlier_multi_columns(self, data, methods_columns_dict):
        """
        Apply different outlier removal or impact reducing methods to specified columns.

        Args:
            data (pd.DataFrame): Input data.
            methods_columns_dict (dict): Dictionary where keys are method names and values are lists of column names.

        Returns:
            pd.DataFrame: Processed data after applying the methods.
        """
        data_copy = data.copy()

        for method, columns in methods_columns_dict.items():
            if method not in self.get_available_methods():
                raise ValueError(f"Unknown method '{method}'")
            
            if method in ["zscore", "yeo_johnson"]:
                raise ValueError(f"Method '{method}' may remove rows. Not recommended for multi-column processing.")
            
            if not isinstance(columns, list):
                raise ValueError(f"Expected a list of columns for method '{method}', got {type(columns)} instead.")

            outlier_remover = OutlierRemover(method=method, threshold=self.threshold, smooth_factor=self.smooth_factor)

            for col in columns:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in data.")
                data_copy[col] = outlier_remover.fit_transform(data[[col]]).iloc[:, 0]

        return data_copy
    
    def get_available_methods(self):
        """
        Returns the available methods for outlier removal.

        Returns:
            list: List of supported methods.
        """
        return [
            "zscore",
            "zscore_capper",
            "yeo_johnson",
            "yeo_johnson_capper",
            "impact_reduction",
            "adaptive_trimming",
            "smooth_capping",
            "local_standardization"
        ]
        
    
    def _zscore_removal(self,data:pd.DataFrame)->pd.DataFrame:
        """
        Remove rows based on Z-score threshold.
        """
        
        z_scores=np.abs(stats.zscore(data.select_dtypes(include=[np.number])))  
        filtered_entries=(z_scores<self.threshold).all(axis=1)
        return data[filtered_entries]
    
    def _zscore_capper(self, data:pd.DataFrame)->pd.DataFrame:
        """
        Cap values based on Z-score threshold.
        """
        data_copy = data.copy()
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_zscore = stats.zscore(data_copy[col])
            data_copy[col] = np.where(
                col_zscore > self.threshold,
                data_copy[col].mean() + self.threshold * data_copy[col].std(),
                np.where(
                    col_zscore < -self.threshold,
                    data_copy[col].mean() - self.threshold * data_copy[col].std(),
                    data_copy[col]
                )
            )
        return data_copy
    
    def _yeo_johnson_removal(self,data:pd.DataFrame)->pd.DataFrame:
        """
        Apply Yeo-Johnson transformation and remove outliers based on z-scores.
        """
        pt=PowerTransformer(method='yeo-johnson')
        numeric_cols=data.select_dtypes(include=[np.number]).columns
        transformed=pt.fit_transform(data[numeric_cols])
        z_scores=np.abs(stats.zscore(transformed))
        data_copy=data.copy()
        data_copy[numeric_cols]=transformed
        filtered_entries=(z_scores<self.threshold).all(axis=1)
        return data_copy[filtered_entries]
    
    def _yeo_johnson_capper(self, data:pd.DataFrame)->pd.DataFrame:
        """
        Apply Yeo-Johnson transformation and cap outliers.
        """
        pt = PowerTransformer(method='yeo-johnson')
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        transformed = pt.fit_transform(data[numeric_cols])
        capped = pd.DataFrame(transformed, columns=numeric_cols)

        for col in numeric_cols:
            capped[col] = np.where(
                capped[col] > self.threshold,
                self.threshold,
                np.where(
                    capped[col] < -self.threshold,
                    -self.threshold,
                    capped[col]
                )
            )

        data_copy = data.copy()
        data_copy[numeric_cols] = capped
        return data_copy
    
    def _reduce_impact(self,data:pd.DataFrame)->pd.DataFrame:
        """
        Reduce impact of outliers by capping values at a specified threshold.
        """
        data_copy=data.copy()
        numeric_cols=data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean=data_copy[col].mean()
            std=data_copy[col].std()
            upper=mean+self.threshold*std
            lower=mean-self.threshold*std
            data_copy[col]=np.where(data_copy[col]>upper,upper,np.where(data_copy[col]<lower,lower,data_copy[col]))
            
        return data_copy
    
    def _adaptive_trimming(self,data:pd.DataFrame)->pd.DataFrame:
        """
        Trim outliers using the IQR method and replace with mean.
        """
        data_copy=data.copy()
        numeric_cols=data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1=data_copy[col].quantile(0.25)
            q3=data_copy[col].quantile(0.75)
            iqr=q3-q1
            lower_bound=q1-1.5*iqr
            upper_bound=q3+1.5*iqr
            mean_val=data_copy[col].mean()
            data_copy[col]=data_copy[col].apply(lambda x: mean_val if x<lower_bound or x>upper_bound else x)
        return data_copy        
    
    def _smooth_boundary_capping(self, data:pd.DataFrame, smooth_factor=None)->pd.DataFrame:
        """
        Softly cap values towards the boundary instead of hard-clipping.

        Args:
            data (pd.DataFrame): Input dataset.
            smooth_factor (float): Smoothing factor (between 0 and 1).
                                Example: 0.9 means pull 90% toward boundary.

        Returns:
            pd.DataFrame: Data after soft boundary capping.
        """
        if smooth_factor is None:
            smooth_factor=self.smooth_factor
        data_copy = data.copy()
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            mean = data_copy[col].mean()
            std = data_copy[col].std()
            upper = mean + self.threshold * std
            lower = mean - self.threshold * std

            def cap_value(x):
                if x < lower:
                    return lower + smooth_factor * (x - lower)  
                elif x > upper:
                    return upper + smooth_factor * (x - upper)  
                else:
                    return x

            data_copy[col] = data_copy[col].apply(cap_value)

        return data_copy

    
    def _local_standardization_correction(self,data:pd.DataFrame,window:int=5)->pd.DataFrame:
        """
        Apply local standardization correction using a rolling window.
        """
        data_copy=data.copy()
        numeric_cols=data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series=data_copy[col]
            local_means=series.rolling(window,center=True,min_periods=1).mean()
            local_stds=series.rolling(window,center=True,min_periods=1).std().fillna(0)
            
            upper_bound=local_means+self.threshold*local_stds
            lower_bound=local_means-self.threshold*local_stds 
            
            data_copy[col]=np.where(series>upper_bound,upper_bound,np.where(series<lower_bound,lower_bound,series)) 
               
        return data_copy   
    def __repr__(self):
        return f"OutlierRemover(method={self.method}, threshold={self.threshold}, smooth_factor={self.smooth_factor})"    

       