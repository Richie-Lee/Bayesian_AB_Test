import random
import numpy as np
import pandas as pd
from datetime import time

def get_bernoulli_sample(mean, n):
    # Sample bernoulli distribution with relevant metrics
    samples = [1 if random.random() < mean else 0 for _ in range(n)]
    converted = sum(samples)
    mean = converted/n 

    return samples, converted, mean

def get_normal_sample(mean, variance, n):
    sample = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    return sample



class get_real_data():
    
    def __init__(self, data_config, simulated_treatment_effect, SEED):
        # Import dataset from local CSV file
        self.df_raw = pd.read_csv(data_config["import_directory"]) 
        
        self.data_config = data_config
        self.treatment = simulated_treatment_effect
        
        self.SEED = SEED
        
        # Execute main method
        self.get_values()

    def start_time(self, df, data_config, df_raw):
        # Skip if no custom start-time is provided (default 00:00)
        if data_config["start_time_hour"] == 0 and data_config["start_time_minute"] == 0:
            return df
        
        # Slice dataset s.t. it starts at specified time
        df["time"] = pd.to_datetime(df[data_config["time_variable"]]).dt.time # parse time strings to time object
        start_time = time(data_config["start_time_hour"], data_config["start_time_minute"])
        df = df[df["time"] > start_time]
        
        # Print new start-time and number of samples that are sliced away
        print(f"Real_data: {data_config['import_directory']}\nstart time: {start_time}, sample remaining: {len(df)}/{len(df_raw)} ({round(len(df)/len(df_raw), 3) * 100}%)")
        return df
    
    def apply_treatment_effect(self, df, voi, treatment):
        # probability of being assigned treatment
        treatment_proportion = 0.5
        
        # Generate random assignments for each user (A/B split randomisation - sample ratio mismatch can be there)
        df['group'] = np.random.choice(['control', 'treatment'], size=len(df), p=[1 - treatment_proportion, treatment_proportion])
        
        # apply treatment effect (either relative or absolute)
        if treatment["relative_treatment_effect"] != None and treatment["absolute_treatment_effect"] == None: 
            df.loc[df['group'] == 'treatment', voi] *= treatment["relative_treatment_effect"]
        elif treatment["relative_treatment_effect"] == None and treatment["absolute_treatment_effect"] != None:
            absolute_effect = treatment["absolute_treatment_effect"]
            df.loc[df['group'] == 'treatment', voi] += absolute_effect
        else:
            raise Exception("Provide EITHER a relative or absolute treatment effect, the other (not-used) variant should be specified as None")
        
        return df
    
    def sample_ratio_mismatch(self, df):
        # # Display the distribution of users in each group
        # group_counts_before = df['group'].value_counts()
        # print(group_counts_before)
        
        # Calculate the size of the smaller group
        group_counts_before = df['group'].value_counts()
        min_group_size = group_counts_before.min()
        
        # Create DataFrames for each group
        control_group = df[df['group'] == 'control']
        treatment_group = df[df['group'] == 'treatment']
        
        # Randomly sample observations from the larger group to match the size of the smaller group
        if group_counts_before['control'] > min_group_size:
            control_group = control_group.sample(n=min_group_size, random_state=42)
        
        if group_counts_before['treatment'] > min_group_size:
            treatment_group = treatment_group.sample(n=min_group_size, random_state=42)
        
        # Concatenate the balanced DataFrames back together
        df = pd.concat([control_group, treatment_group])
        
        # Sort by time for chronological data (At this point, it should already be in datetime format)
        df = df.sort_values(by = "time", ascending = True)
        
        # # Display the distribution of users in each group (they should now have the same size)
        # group_counts_after = df['group'].value_counts()
        # print(group_counts_after)
        
        return df
        
    def format_to_array(self, df, voi, data_config, SEED):
        # Get data per split
        data_C_df = df[df['group'] == 'control']
        data_T_df = df[df['group'] == 'treatment']
        
        # samples (df -> np array)
        C_sample = data_C_df[voi].to_numpy()
        T_sample = data_T_df[voi].to_numpy()
        
        
        C = {"n": len(C_sample), "sample": C_sample, "true_mean": C_sample.mean(), "true_variance": C_sample.var(), "df": data_C_df}
        T = {"n": len(T_sample), "sample": T_sample, "true_mean": T_sample.mean(), "true_variance": T_sample.var(), "df": data_T_df}
        
        # for universal naming, create voi column for voi 
        df["voi"] = df[voi]
        
        return df, C, T
        
    
    def get_values(self):
        df = self.df_raw.copy()
        voi = self.data_config["voi"]
        
        df = self.start_time(df, self.data_config, self.df_raw)
        df = self.apply_treatment_effect(df, voi, self.treatment)
        df = self.sample_ratio_mismatch(df)
        
        df, C, T = self.format_to_array(df, voi, self.data_config, self.SEED)
        return C, T, df, voi