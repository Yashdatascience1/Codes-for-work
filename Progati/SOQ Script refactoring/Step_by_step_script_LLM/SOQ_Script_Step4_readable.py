import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import math
import numpy as np
import pandas as pd
import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

MONTHS          = ['2026-06-01']     # planning months to process
STOCK_DATE_TYPE = ['first']          # stock snapshot type: "first", "end", or "mid"
RUN_VERSION     = 33
RUN_DATE        = datetime.datetime.today().strftime('%Y%m%d')
IS_OBD          = True
obd_flag        = 'Y' if IS_OBD else 'N'

# Safety stock days per ABC tier.
# A-class = high-value/fast-moving -> more buffer (30 days)
# C-class = low-value/slow-moving  -> less buffer (20 days)
ABC = {"A": 30, "B": 25, "C": 20}

# This string is stored on every output row as an audit trail of what ABC
# thresholds were used in this run (e.g. "A 30,B 25,C 20")
STR_ABC = ','.join([f"{k} {v}" for k, v in ABC.items()])

# Z-scores for one-tailed normal distribution at various service levels.
# Higher service level = higher Z = more safety stock.
Z_SCORE = {95: 1.65, 90: 1.28, 85: 1.04, 80: 0.85, 99: 2.33}

# Table paths
BASE_SOQ_TABLE               = 'MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED'
DEMAND_VARIABILITY_TABLE     = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'
DEMAND_VARIABILITY_FAMILY_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'
SOQ_TABLE                    = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'
NULL_LEAD_TIME_TABLE         = 'MOP_DATABASE.SOQ.NULL_LEAD_TIME'

END_JOURNEY_QUERY = """
    SELECT SKU, RECOMMEND_END_OF_JOURNEY
    FROM MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION
    WHERE RUN_DATE = (SELECT MAX(RUN_DATE) FROM MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION)
"""


def main(session: snowpark.Session):

    # =========================================================================
    # OUTER LOOP: one iteration per planning month × stock period × service level
    # × demand variability mode.
    #
    # Running all 10 combinations (5 service levels × 2 DV modes) in a single
    # script means the output table captures every scenario in one pipeline run.
    # =========================================================================

    service_levels        = [95, 90, 85, 99, 80]
    demand_variability_modes = [True, False]   # True=SKU-level, False=Family-level

    for planning_month in MONTHS:
        for date_period in STOCK_DATE_TYPE:
            for service_level in service_levels:
                for sku_demand_variability in demand_variability_modes:

                    print(f"\n=== Month: {planning_month} | Period: {date_period} | "
                          f"SL: {service_level}% | DV: {'SKU' if sku_demand_variability else 'FAMILY'} ===")


                    # =============================================================
                    # BLOCK 1 — LOAD THE SOQ BASE DATA
                    #
                    # The SOQ base table (produced in Step 3) has one row per
                    # dealer + SKU + planning month. It contains:
                    #   - PREDICTED_SALES_SKU  : SKU-level forecast from the TFT model
                    #   - STK_AS_ON_DATE       : physical stock at the snapshot date
                    #   - MAX/MIN/AVG_LEAD_TIME: transit days from Step 1 transit pipeline
                    #   - UNIQUE FAMILY CODE   : product family grouping
                    #
                    # We filter to a specific run slice (month + period + run_date +
                    # version + OBD flag) to ensure repeatability.
                    # =============================================================

                    soq_query = f"""
                        SELECT * FROM {BASE_SOQ_TABLE}
                        WHERE PLANNING_MONTH = '{planning_month}'
                          AND STOCK_DATE_PERIOD = '{date_period}'
                          AND RUN_DATE = {RUN_DATE}
                          AND RUN_VERSION = {RUN_VERSION}
                          AND IS_OBD = '{obd_flag}'
                    """
                    soq_data = session.sql(soq_query).to_pandas()
                    # Drop IS_OBD — it's a filter key, not a useful output column
                    soq_data.drop(columns=['IS_OBD'], inplace=True)
                    print(f"SOQ base rows loaded: {len(soq_data)}")


                    # =============================================================
                    # BLOCK 2 — LOAD AND MERGE DEMAND VARIABILITY
                    #
                    # Demand variability (std dev of month-over-month sales deltas)
                    # was pre-computed in Step 3. It tells us how erratic each
                    # dealer-SKU series is — the more erratic, the more safety stock.
                    #
                    # Two modes:
                    #   SKU-level   (sku_demand_variability=True):
                    #     Join on PARENT_DEALER_CODE + UNIQUE FAMILY CODE + SKU.
                    #     More granular — each SKU gets its own variability number.
                    #
                    #   Family-level (sku_demand_variability=False):
                    #     Join on PARENT_DEALER_CODE + UNIQUE FAMILY CODE only.
                    #     Uses the family's aggregate variability for all its SKUs.
                    #     Used when a SKU doesn't have enough individual history.
                    # =============================================================

                    if sku_demand_variability:
                        dv_query = f"""
                            SELECT * FROM {DEMAND_VARIABILITY_TABLE}
                            WHERE PLANNING_MONTH = '{planning_month}'
                              AND RUN_DATE = {RUN_DATE}
                              AND RUN_VERSION = {RUN_VERSION}
                              AND IS_OBD = '{obd_flag}'
                        """
                        demand_variability = session.sql(dv_query).to_pandas()
                        demand_variability.drop(columns=['IS_OBD'], inplace=True)

                        data = pd.merge(
                            soq_data, demand_variability,
                            on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'SKU', 'PLANNING_MONTH'],
                            how='left'
                        )
                        data['DEMAND_VARIABILITY_TYPE'] = 'SKU_BASED'

                    else:
                        dv_query = f"""
                            SELECT * FROM {DEMAND_VARIABILITY_FAMILY_TABLE}
                            WHERE PLANNING_MONTH = '{planning_month}'
                              AND RUN_DATE = {RUN_DATE}
                              AND RUN_VERSION = {RUN_VERSION}
                              AND IS_OBD = '{obd_flag}'
                        """
                        demand_variability = session.sql(dv_query).to_pandas()
                        demand_variability.drop(columns=['IS_OBD'], inplace=True)

                        data = pd.merge(
                            soq_data, demand_variability,
                            on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PLANNING_MONTH'],
                            how='left'
                        )
                        data['DEMAND_VARIABILITY_TYPE'] = 'MODEL_SKU_FAMILY_BASED'

                    # Any SKU that couldn't be matched gets variability = 1 (neutral default).
                    # This means safety stock = Z * sqrt(days/30), with no demand signal.
                    data.loc[data['DEMAND_VARIABILITY'].isnull(), 'DEMAND_VARIABILITY'] = 1


                    # =============================================================
                    # BLOCK 3 — ABC CLASSIFICATION
                    #
                    # ABC classifies each (dealer, family) by its share of the
                    # dealer's total forecasted sales, using cumulative contribution.
                    #
                    # Steps:
                    #   1. Deduplicate to one row per dealer + family
                    #      (data is at SKU grain, but ABC is at family grain)
                    #   2. Compute each family's % of total dealer sales
                    #   3. Sort high-to-low, cumsum the percentages
                    #   4. A = top 70% of revenue (fast-movers)
                    #      B = 70–90%
                    #      C = bottom 10% (slow-movers)
                    #   5. Map ABC tier to safety stock days from config
                    # =============================================================

                    # Step 1: one row per dealer + family for the ranking
                    sales_data = (
                        data[['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PREDICTED_SALES']]
                        .drop_duplicates()
                        .copy()
                    )

                    # Step 2: dealer total (denominator)
                    dealer_totals = (
                        sales_data.groupby('PARENT_DEALER_CODE')['PREDICTED_SALES']
                        .sum().reset_index()
                        .rename(columns={'PREDICTED_SALES': 'DEALER_PREDICTED_SALES'})
                    )
                    sales_data = pd.merge(sales_data, dealer_totals,
                                          on='PARENT_DEALER_CODE', how='left')

                    # Step 3: percentage of dealer total, handle zero-total dealers
                    sales_data['PERC_SALES'] = (
                        (sales_data['PREDICTED_SALES'] / sales_data['DEALER_PREDICTED_SALES']) * 100
                    ).fillna(0)

                    # Step 4: sort and cumsum
                    sales_data = sales_data.sort_values(
                        by=['PARENT_DEALER_CODE', 'PERC_SALES'], ascending=[True, False]
                    )
                    sales_data['CUMULATIVE_PERCENT_SALES'] = (
                        sales_data.groupby('PARENT_DEALER_CODE')['PERC_SALES'].cumsum()
                    )

                    # Step 5: assign label
                    def assign_abc(cum_pct):
                        if cum_pct < 70:
                            return 'A'
                        elif cum_pct > 90:
                            return 'C'
                        else:
                            return 'B'

                    sales_data['ABC'] = sales_data['CUMULATIVE_PERCENT_SALES'].apply(assign_abc)
                    sales_data = sales_data[['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'ABC']]

                    # Join ABC back onto the SKU-level data
                    data = pd.merge(data, sales_data,
                                    on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE'], how='left')

                    # Map ABC to safety stock days (default 15 if unclassified)
                    data['SAFETY_STOCK_DAYS'] = data['ABC'].apply(lambda x: ABC.get(x, 15))


                    # =============================================================
                    # BLOCK 4 — HANDLE NULL LEAD TIMES
                    #
                    # Rows with no transit data (MAX_LEAD_TIME is null) cannot
                    # get a valid SOQ. Save them to a separate audit table so
                    # the operations team can investigate missing transit records,
                    # then exclude them from the main calculation.
                    # =============================================================

                    null_lt_rows = data[data['MAX_LEAD_TIME'].isnull()]
                    if len(null_lt_rows) > 0:
                        session.create_dataframe(null_lt_rows).write.mode("overwrite").save_as_table(
                            NULL_LEAD_TIME_TABLE
                        )
                        print(f"  {len(null_lt_rows)} rows with null lead time saved to {NULL_LEAD_TIME_TABLE}")

                    data = data[~data['MAX_LEAD_TIME'].isnull()].copy()


                    # =============================================================
                    # BLOCK 5 — LEAD TIME STOCK
                    #
                    # How many units are sold while waiting for the delivery to arrive?
                    # Formula: LEAD_TIME_STOCK = ceil( PREDICTED_SALES_SKU * LEAD_DAYS / 30 )
                    #
                    # Dividing by 30 converts monthly forecast to a daily rate.
                    # Multiplying by LEAD_DAYS gives units consumed during transit.
                    # ceil() ensures we never under-order due to rounding.
                    #
                    # Computed for MAX, MIN, and AVG lead time variants so planners
                    # can choose between conservative (MAX) and optimistic (MIN) SOQs.
                    # =============================================================

                    for variant in ['MAX', 'MIN', 'AVG']:
                        data[f'{variant}_LEAD_TIME_STOCK'] = (
                            (data['PREDICTED_SALES_SKU'] * data[f'{variant}_LEAD_TIME']) / 30
                        ).apply(math.ceil)


                    # =============================================================
                    # BLOCK 6 — SAFETY STOCK
                    #
                    # Safety stock is the buffer held to protect against demand
                    # uncertainty. Formula:
                    #
                    #   SAFETY_STOCK = ceil( DEMAND_VARIABILITY
                    #                        * sqrt( SAFETY_STOCK_DAYS / 30 )
                    #                        * Z_SCORE )
                    #
                    # Components:
                    #   DEMAND_VARIABILITY  = std dev of MoM deltas (how erratic is demand)
                    #   sqrt(days/30)       = scaled to a monthly fraction (days of coverage)
                    #   Z_SCORE             = confidence multiplier for desired service level
                    #
                    # Hard cap at 3× monthly predicted sales.
                    # Without this cap, a single anomalous month of high variability
                    # could produce an absurdly large safety stock for a low-volume SKU.
                    # =============================================================

                    z = Z_SCORE[service_level]
                    raw_ss = data['DEMAND_VARIABILITY'] * np.sqrt(data['SAFETY_STOCK_DAYS'] / 30)
                    data['SAFETY_STOCK'] = raw_ss.apply(lambda x: math.ceil(x * z))

                    # Cap: safety stock cannot exceed 3 months of predicted sales
                    data['SAFETY_STOCK'] = np.minimum(
                        data['SAFETY_STOCK'].fillna(0),
                        data['PREDICTED_SALES_SKU'].fillna(0) * 3
                    )


                    # =============================================================
                    # BLOCK 7 — SOQ COMPUTATION: TWO APPROACHES
                    #
                    # Fill null stock with 0 (dealer has no inventory on hand).
                    #
                    # APPROACH 1 — Full coverage:
                    #   REORDER_STOCK   = SAFETY_STOCK + LEAD_TIME_STOCK
                    #   TOTAL_NEED      = PREDICTED_SALES + REORDER_STOCK
                    #                     (covers both sales AND the safety buffer)
                    #   SUGGESTED_ORDER = TOTAL_NEED - CURRENT_STOCK
                    #   ADJUSTED_ORDER  = max(0, SUGGESTED_ORDER)
                    #                     (can never order a negative quantity)
                    #
                    # APPROACH 2 — Reorder point only:
                    #   SOQ             = REORDER_STOCK - CURRENT_STOCK
                    #   ADJUSTED_ORDER  = max(0, SOQ)
                    #   (Assumes forecasted sales will be covered separately.
                    #    Only fills up to the safety buffer level.)
                    #
                    # Both approaches are computed for MAX, MIN, and AVG lead times.
                    # =============================================================

                    data.loc[data['STK_AS_ON_DATE'].isnull(), 'STK_AS_ON_DATE'] = 0

                    for variant in ['MAX', 'MIN', 'AVG']:
                        lt   = data[f'{variant}_LEAD_TIME_STOCK']
                        ss   = data['SAFETY_STOCK']
                        pred = data['PREDICTED_SALES_SKU']
                        stk  = data['STK_AS_ON_DATE']

                        # Approach 1
                        reorder  = ss + lt
                        total    = pred + reorder
                        suggested = total - stk

                        data[f'{variant}_REORDER_STOCK']          = reorder
                        data[f'{variant}_TOTAL_STOCK_SKU']        = total
                        data[f'{variant}_Suggested_Stock_SKU']    = suggested
                        data[f'{variant}_Adjusted_Monthly_Order'] = suggested.apply(
                            lambda x: 0 if x < 0 else x
                        )

                        # Approach 2
                        soq2 = reorder - stk
                        data[f'{variant}_SOQ_APPROACH_2']                     = soq2
                        data[f'{variant}_Adjusted_Monthly_Order_APPROACH_2']  = soq2.apply(
                            lambda x: 0 if x < 0 else x
                        )


                    # =============================================================
                    # BLOCK 8 — METADATA + END-OF-JOURNEY FLAG
                    #
                    # Overwrite the ABC column with the human-readable config string
                    # (e.g. "A 30,B 25,C 20") so reviewers know the thresholds used.
                    #
                    # END_OF_JOURNEY_RECOMMENDATION flags SKUs that the business
                    # recommends phasing out. The planner can use this to decide
                    # whether to order at all for that SKU.
                    # =============================================================

                    data['ABC']           = STR_ABC
                    data['Z_SCORE']       = Z_SCORE[service_level]
                    data['SERVICE_LEVEL'] = service_level
                    data['RUN_DATE']      = RUN_DATE
                    data['RUN_VERSION']   = RUN_VERSION
                    data['IS_OBD']        = obd_flag

                    # Fetch the most recently generated end-of-journey flags
                    end_journey = session.sql(END_JOURNEY_QUERY).to_pandas()
                    data = pd.merge(data, end_journey, on='SKU', how='left')


                    # =============================================================
                    # BLOCK 9 — SAVE
                    # Append this scenario's results to the SOQ output table.
                    # Using "append" (not "overwrite") so all 10 scenario
                    # combinations accumulate in one table for comparison.
                    # =============================================================

                    session.create_dataframe(data).write.mode("append").save_as_table(SOQ_TABLE)
                    print(f"  Written {len(data)} rows to {SOQ_TABLE}")

    return session.table(SOQ_TABLE)
