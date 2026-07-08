#simulate_paths.py
# Pathwise Monte Carlo implementation with bounded memory usage.
import numpy as np

# 6m increments, 3y total time
deltat = 6 / 12
startTime = 0.0
endTime = 3.0
ti = np.arange(startTime, endTime + deltat, deltat)

# Coupon dates (semiannual). On a 6m grid this is every time step except t = 0.

RecoveryRate = 0.4
Notional = 100.0
couponRate = 0.03        # 3% per coupon, 6% annually

# Discount curve parameters
a_r = 0.02
b_r = 0.02
c_r = 0.35

D0 = lambda t: np.exp(-a_r * t - (b_r / c_r) * (1.0 - np.exp(-c_r * t)))
DiscountFactors = D0(ti)

# Investment Grade parameters
a_h_ig = 0.015
b_h_ig = -0.012
c_h_ig = 0.3
sigma_h_ig = 0.025

S0_ig = lambda t: np.exp(-a_h_ig * t - (b_h_ig / c_h_ig) * (1.0 - np.exp(-c_h_ig * t)))
survivalProbsIG = S0_ig(ti)

# High Yield parameters
a_h_hy = 0.03
b_h_hy = -0.02
c_h_hy = 0.4
sigma_h_hy = 0.05

S0_hy = lambda t: np.exp(-a_h_hy * t - (b_h_hy / c_h_hy) * (1.0 - np.exp(-c_h_hy * t)))
survivalProbsHY = S0_hy(ti)

# Discount curve parameters
a_r = 0.02
b_r = 0.02
c_r = 0.35

D0 = lambda t: np.exp(-a_r * t - (b_r / c_r) * (1.0 - np.exp(-c_r * t)))
DiscountFactors = D0(ti)

# Investment Grade parameters
a_h_ig = 0.015
b_h_ig = -0.012
c_h_ig = 0.3
sigma_h_ig = 0.025

S0_ig = lambda t: np.exp(-a_h_ig * t - (b_h_ig / c_h_ig) * (1.0 - np.exp(-c_h_ig * t)))
survivalProbsIG = S0_ig(ti)

# High Yield parameters
a_h_hy = 0.03
b_h_hy = -0.02
c_h_hy = 0.4
sigma_h_hy = 0.05

S0_hy = lambda t: np.exp(-a_h_hy * t - (b_h_hy / c_h_hy) * (1.0 - np.exp(-c_h_hy * t)))
survivalProbsHY = S0_hy(ti)

# Initial Survival Probability Curve and Forward Hazard Curve
ForwardHazardsIG = -np.log(survivalProbsIG[1:] / survivalProbsIG[:-1]) / deltat
ForwardHazardsHY = -np.log(survivalProbsHY[1:] / survivalProbsHY[:-1]) / deltat


def vanilla_cln_analytical(S0_function, D0_function, ti, Notional, couponRate, RecoveryRate):

    # Survival and discount curves on the grid
    survivals = S0_function(ti)       
    discounts = D0_function(ti)      

    S_previous = survivals[:-1]       # S(0, t_{k-1})
    S_current = survivals[1:]        # S(0, t_k)
    D_current = discounts[1:]        # D(0, t_k)

    # There is no coupon at time 0
    couponPV = Notional * couponRate * np.sum(D_current * S_current)

    # redemption paid contigent on surviving to maturity
    redemptionPV = Notional * discounts[-1] * survivals[-1]

    # recovery is the same as before
    recoveryPV = Notional * RecoveryRate * np.sum(D_current * (S_previous - S_current))

    return couponPV + redemptionPV + recoveryPV

pvAnalyticalVanillaHY = vanilla_cln_analytical(S0_hy, D0, ti, Notional, couponRate, RecoveryRate)
pvAnalyticalVanillaIG = vanilla_cln_analytical(S0_ig, D0, ti, Notional, couponRate, RecoveryRate)

def simulatePaths(normalVariates, useRangeAccrual):
    n,_ = normalVariates.shape

    # Range accrual condition
    upperBarrier = 150 / 10_000
    lowerBarrier = 0.0

    # Number of semiannual hazard intervals on the simulation grid.
    M = len(ForwardHazardsHY)

    sigma_hat_ig = lambda t1, t2: sigma_h_ig * np.exp(-c_h_ig * (t2 - t1))
    sigma_hat_hy = lambda t1, t2: sigma_h_hy * np.exp(-c_h_hy * (t2 - t1))

    # Pathwise PV arrays
    PV_paths_IG = np.zeros(n)
    PV_paths_HY = np.zeros(n)

    # Monte Carlo (pathwise and memory-safe)
    for sim in range(n):

        # Clone the forward curves
        FWD_IG = ForwardHazardsIG + 0
        FWD_HY = ForwardHazardsHY + 0

        # Each path evolves its own copy of the forward hazard curve.
        # Adding zero forces NumPy to allocate a fresh array.

        accumHazardIG = 1.0
        accumHazardHY = 1.0

        rangeCountIG = 0
        rangeCountHY = 0

        # Coupons are paid at each grid point after t = 0.
        couponPV_IG = 0.0
        couponPV_HY = 0.0
        recoveryPV_IG = 0.0
        recoveryPV_HY = 0.0

        prevSurvIG = 1.0
        prevSurvHY = 1.0

        for ix in range(M):

            # The diagonal entries of the forward curve are the instantaneous hazard rates.
            hazardIG = FWD_IG[ix]
            hazardHY = FWD_HY[ix]

            # clip the hazards to prevent negative spreads
            spreadIG = max(hazardIG, 0.0) * (1.0 - RecoveryRate)
            spreadHY = max(hazardHY, 0.0) * (1.0 - RecoveryRate)

            # Range accrual condition
            if useRangeAccrual:
                if lowerBarrier <= spreadIG < upperBarrier:
                    rangeCountIG += 1
                if lowerBarrier <= spreadHY < upperBarrier:
                    rangeCountHY += 1
            else: 
                # In the vanilla case the accrual fraction is always 1.
                rangeCountIG += 1
                rangeCountHY += 1

            # Survival update
            accumHazardIG *= np.exp(-hazardIG * deltat)
            accumHazardHY *= np.exp(-hazardHY * deltat)
            survIG = accumHazardIG
            survHY = accumHazardHY

            # Every period after t=0 has a coupon since the grid aligns with the schedule
            
            # Discount factors are defined at the start of the interval
            # D(ix) = 1 when ix = 0 i.e. D(0) is discount at today
            # Hence each cash-flow index is shifted by one grid point.
            couponPV_IG += Notional * couponRate * DiscountFactors[ix+1] * survIG
            couponPV_HY += Notional * couponRate * DiscountFactors[ix+1] * survHY

            # Recovery leg
            # shift ix by one, same reason as for coupons
            # Recovery is paid on the interval default probability.
            recoveryPV_IG += Notional * RecoveryRate * DiscountFactors[ix+1] * (prevSurvIG - survIG)
            recoveryPV_HY += Notional * RecoveryRate * DiscountFactors[ix+1] * (prevSurvHY - survHY)

            prevSurvIG = survIG
            prevSurvHY = survHY

            # This is the forward curve evolution part
            if ix < M - 1:
                ti_min1 = ti[ix]
                tj = ti[ix + 1:-1]

                sigma_j_ig = sigma_hat_ig(ti_min1 * np.ones_like(tj), tj)
                sigma_j_hy = sigma_hat_hy(ti_min1 * np.ones_like(tj), tj)

                ws1_ig = np.cumsum(sigma_j_ig * deltat) ** 2
                ws1_hy = np.cumsum(sigma_j_hy * deltat) ** 2

                ws2_ig = np.concatenate(([0.0], np.cumsum(sigma_j_ig[:-1] * deltat) ** 2))
                ws2_hy = np.concatenate(([0.0], np.cumsum(sigma_j_hy[:-1] * deltat) ** 2))

                mu_j_ig = (ws1_ig - ws2_ig) / (2.0 * deltat)
                mu_j_hy = (ws1_hy - ws2_hy) / (2.0 * deltat)

                Z = normalVariates[sim,ix]

                FWD_IG[ix + 1:] += mu_j_ig * deltat + sigma_j_ig * np.sqrt(deltat) * Z
                FWD_HY[ix + 1:] += mu_j_hy * deltat + sigma_j_hy * np.sqrt(deltat) * Z

        # Redemption leg (range accrual fraction)
        redemptionPV_IG = Notional * (rangeCountIG / M) * DiscountFactors[-1] * survIG
        redemptionPV_HY = Notional * (rangeCountHY / M) * DiscountFactors[-1] * survHY

        PV_paths_IG[sim] = redemptionPV_IG + couponPV_IG + recoveryPV_IG
        PV_paths_HY[sim] = redemptionPV_HY + couponPV_HY + recoveryPV_HY

    return PV_paths_IG, PV_paths_HY, pvAnalyticalVanillaIG, pvAnalyticalVanillaHY
   
