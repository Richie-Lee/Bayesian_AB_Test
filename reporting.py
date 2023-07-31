import numpy as np

def metrics(T, C, prior, hypotheses, n_runs):
    # Evaluate Bayes Factor probability (likelihoods)
    prob_H1_given_BF = round(T["bayes_factor"] / (T["bayes_factor"] + 1), 3)
    
    # Evaluate how often treatment outperformes mde
    treatment_better_mde = [t - c >= hypotheses['mde'] for c, t in zip(C["post_sample"], T["post_sample"])]
    prob_TE_better_mde = round(np.mean(treatment_better_mde), 2)
    
    # Evaluate how often treatment outperformes control
    treatment_won = [t >= c for c, t in zip(C["post_sample"], T["post_sample"])]
    prob_TE_positive = round(np.mean(treatment_won), 2)

    # Get treatment effect measurement
    treatment_effect = {
            "true": round(T["true_prob"] - C["true_prob"], 4),
            "observed": round(T["sample_conversion_rate"] - C["sample_conversion_rate"], 4),
            "estimated": round(T["post_sample"].mean() - C["post_sample"].mean(), 4),
            "prior": round(prior["prior_treatment"] - prior["prior_control"], 4),
            "P[T > C]": prob_TE_positive,
            "P[TE > MDE]": prob_TE_better_mde
        }
    
    # Prevent excessive printing when running multiple times (only print single iterations)
    if n_runs == False:
        print(f"\n===============================\nH0: y = {hypotheses['null']}, H1: y = {hypotheses['alt']}\nmde = {hypotheses['mde']} \n===============================")
        print(f"\nInformal probabilities: \nP[H1|BF]: {prob_H1_given_BF}")
        print(f"posterior: P[T - C >= mde]: {prob_TE_better_mde}")
        print(f"posterior: P[T >= C]: {prob_TE_positive}")
        print(f"\nTreatment effect:\n- true: {treatment_effect['true']}\n- observed: {treatment_effect['observed']}\n- prior: {treatment_effect['prior']}\n- posterior: {treatment_effect['estimated']}")

    return treatment_effect