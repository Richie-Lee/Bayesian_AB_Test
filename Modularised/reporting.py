import numpy as np

def metrics(T, C, prior, hypotheses):
    print(f"\n===============================\nH0: y = {hypotheses['null']}, H1: y = {hypotheses['alt']}\nmde = {hypotheses['mde']} \n===============================")
    
    prob_H1_given_BF = round(T["bayes_factor"] / (T["bayes_factor"] + 1), 3)
    print(f"\nInformal probabilities: \nP[H1|BF]: {prob_H1_given_BF}")

    # Evaluate how often treatment outperformes control
    treatment_won = [t - c >= hypotheses['mde'] for c, t in zip(C["post_sample"], T["post_sample"])]
    prob_TE_better_mde = round(np.mean(treatment_won), 2)
    print(f"posterior: P[T - C >= mde]: {prob_TE_better_mde}")
    treatment_won = [t >= c for c, t in zip(C["post_sample"], T["post_sample"])]
    prob_TE_positive = round(np.mean(treatment_won), 2)
    print(f"posterior: P[T >= C]: {prob_TE_positive}")



    # Get treatment effect measurement
    treatment_effect = {
            "true": round(T["true_prob"] - C["true_prob"], 4),
            "observed": round(T["sample_conversion_rate"] - C["sample_conversion_rate"], 4),
            "estimated": round(T["post_sample"].mean() - C["post_sample"].mean(), 4),
            "prior": round(prior["prior_treatment"] - prior["prior_control"], 4),
            "P[T > C]": prob_TE_positive,
            "P[TE > MDE]": prob_TE_better_mde
        }

    print(f"\nTreatment effect:\n- true: {treatment_effect['true']}\n- observed: {treatment_effect['observed']}\n- prior: {treatment_effect['prior']}\n- posterior: {treatment_effect['estimated']}")

    # # Compute loss (Reward/Penalise not choosing probability closest to the truth, by difference |T-C|)
    # loss_control = [max(j - i, 0) for i,j in zip(C["post_sample"], T["post_sample"])]
    # loss_control = [int(i)*j for i,j in zip(treatment_won, loss_control)]
    # loss_control = round(np.mean(loss_control), 4)

    # loss_treatment = [max(i - j, 0) for i,j in zip(C["post_sample"], T["post_sample"])]
    # loss_treatment = [(1 - int(i))*j for i,j in zip(treatment_won, loss_treatment)]
    # loss_treatment = round(np.mean(loss_treatment), 4)

    # print(f"\nLoss (acceptable <= {round(treatment_effect['prior'] * _relative_loss_theshold, 4)}):\n- Treatment: {loss_treatment}\n- Control: {loss_control}")

    return treatment_effect