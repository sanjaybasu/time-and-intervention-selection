"""
Table generation for population health analysis.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Tables:
    """Generate tables for analysis results."""
    
    def create_performance_table(self, results):
        """Create model performance table."""
        table_data = []
        
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            for horizon in ['7d', '30d', '90d', '180d']:
                if horizon in model_results:
                    auc = model_results[horizon].get('auc', np.nan)
                    auc_ci = model_results[horizon].get('auc_ci', (np.nan, np.nan))
                    
                    row[f'AUC_{horizon}'] = f"{auc:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def create_clinical_utility_table(self, results):
        """Create clinical utility table."""
        table_data = []
        
        for model_name, model_results in results.items():
            if '7d' in model_results:
                nnt = model_results['7d'].get('nnt', np.inf)
                nnt_ci = model_results['7d'].get('nnt_ci', (np.inf, np.inf))
                
                row = {
                    'Model': model_name,
                    'NNT_7d': f"{nnt:.2f} ({nnt_ci[0]:.2f}-{nnt_ci[1]:.2f})",
                    'Clinical_Utility': 'High' if nnt < 5 else 'Moderate' if nnt < 10 else 'Low'
                }
                
                table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def generate_all_tables(self, results, output_dir):
        """Generate all tables."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance table
        perf_table = self.create_performance_table(results)
        perf_table.to_csv(f"{output_dir}/model_performance.csv", index=False)
        
        # Clinical utility table
        utility_table = self.create_clinical_utility_table(results)
        utility_table.to_csv(f"{output_dir}/clinical_utility.csv", index=False)
        
        logger.info(f"Tables saved to {output_dir}")
