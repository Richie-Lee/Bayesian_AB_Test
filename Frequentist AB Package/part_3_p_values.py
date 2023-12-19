from scipy import stats

class get_p_value():
    def __init__(self, T, C, early_stopping_settings):
        self.T = T
        self.C = C
        self.early_stopping_settings = early_stopping_settings

    def t_test(self, T_sample, C_sample):
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(T_sample, C_sample, equal_var=False)
        return p_value / 2  # One-tailed p-value

    def get_values(self):
        n_observed, interim_tests = 0, []
        min_sample = self.early_stopping_settings["minimum_sample"]

        # Fixed Horizon
        p_value_fixed_horizon = self.t_test(self.T["sample"], self.C["sample"])

        # Early Stopping
        while n_observed <= len(self.T["sample"]):
            T_sample = self.T["sample"][:n_observed]
            C_sample = self.C["sample"][:n_observed]
            p_value = self.t_test(T_sample, C_sample)
            interim_tests.append((n_observed, p_value))

            if n_observed >= min_sample:
                if p_value < self.early_stopping_settings["alpha"] or n_observed == len(self.T["sample"]):
                    break

            n_observed += self.early_stopping_settings["interim_test_interval"]

        return p_value_fixed_horizon, p_value, interim_tests, n_observed
