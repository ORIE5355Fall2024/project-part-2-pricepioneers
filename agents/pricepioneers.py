import random
import pickle
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog


'''
This template serves as a starting point for your agent.
'''


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## for example, we will run with inventory_replenish = 20, with the limit of 11. Then, the inventory will be replenished every 20 time steps (time steps 0, 20, 40, ...) and the inventory will be set to 11 at those time steps. 
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        ### useful if you want to use a more complex price prediction model
        ### note that you will need to change the name of the path and this agent file when submitting
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### however, this does not work with custom defined classes, due to the way pickle serializes objects
        ### refer to './yourteamname/create_model.ipynb' for a quick tutorial on how to use pickle
        # self.filename = './[yourteamname]/trained_model'
        # self.trained_model = pickle.load(open(self.filename, 'rb'))

        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        # self.opponent_number = 1 - agent_number  # index for opponent

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who, if anyone, the customer bought from
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        ### potentially useful for Part 2
        # did_customer_buy_from_opponent = (last_sale[0] == self.opponent_number)

        ### keep track of the prices that were offered in the last sale
        my_last_prices = last_sale[1][self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_last_prices = last_sale[1][self.opponent_number]

        ### keep track of the profit for this agent after the last sale
        my_current_profit = state[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_current_profit = state[self.opponent_number]

        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_inventory = inventories[self.opponent_number]

        ### keep track of the time until the next replenishment
        time_until_replenish = time_until_replenish

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round
        pass

    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.
        '''

        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        # Getting the inventory level of our agent (assumes inventories is a vector, so we take the first value)
        inventory_level = inventories[0]

        train_pricing_decisions = pd.read_csv('agents/pricepioneers/train_prices_decisions_2024.csv')
        
        min_price_threshold = np.percentile(train_pricing_decisions['price_item'], 10)
        prices_to_predict = np.arange(min_price_threshold, train_pricing_decisions['price_item'].max() + train_pricing_decisions['price_item'].mean(), 4)

        with open('agents/pricepioneers/randomforrest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)

        # Helper functions within action:
        def predict_optimal_price(df, prices_to_predict, rf_model):
            expanded_covariates = pd.DataFrame(np.tile(df[['Covariate1', 'Covariate2', 'Covariate3']].values, (len(prices_to_predict), 1)),
                                            columns=['Covariate1', 'Covariate2', 'Covariate3'])

            expanded_prices = np.repeat(prices_to_predict, len(df))
            expanded_data = pd.DataFrame({
                'price_item': expanded_prices,
                'Covariate1': expanded_covariates['Covariate1'],
                'Covariate2': expanded_covariates['Covariate2'],
                'Covariate3': expanded_covariates['Covariate3']
            })
            predictions = rf_model.predict_proba(expanded_data)[:, 1]  
            predictions_matrix = predictions.reshape(len(df), len(prices_to_predict))
            revenues_matrix = predictions_matrix * prices_to_predict
            max_revenue_prices = prices_to_predict[np.argmax(revenues_matrix, axis=1)]
            df['predicted_price'] = max_revenue_prices
            demand_prediction_df = pd.DataFrame(predictions_matrix, columns=prices_to_predict)
            return df, demand_prediction_df

        def get_single_step_revenue_maximizing_price_and_revenue_k(Vtplus1k, Vtplus1kminus1, price_options, demand_predictions):
            price_options = np.array(price_options, dtype=np.float64)
            demand_predictions = np.array(demand_predictions, dtype=np.float64)
            rev_list = (price_options + Vtplus1kminus1) * demand_predictions + (1 - demand_predictions) * Vtplus1k
            opt_index = np.argmax(rev_list)
            return price_options[opt_index], rev_list[opt_index]

        def get_prices_over_time_and_expected_revenue_k(prices, demand_predictions, T, K):
            prices = np.array(prices, dtype=np.float64)
            demand_predictions = np.array(demand_predictions, dtype=np.float64)
            opt_price_list = np.zeros((T, K + 1), dtype=np.float64)
            V = np.zeros((T + 1, K + 1), dtype=np.float64)

            for t in range(T - 1, -1, -1):
                V_t_k = V[t + 1, 1:]
                V_t_k_minus_1 = V[t + 1, :-1]
                rev_list = (prices + V_t_k_minus_1[:, None]) * demand_predictions + (1 - demand_predictions) * V_t_k[:, None]
                opt_index = np.argmax(rev_list, axis=1)
                opt_prices = prices[opt_index]
                max_values = np.max(rev_list, axis=1)
                V[t, 1:] = max_values
                opt_price_list[t, 1:] = opt_prices
            return opt_price_list, V

        # Load threshold data (these are the thresholds you need)
        threshold_10percentile = pd.read_csv('agents/pricepioneers/threshold_10percentile.csv')
        threshold_avg = pd.read_csv('agents/pricepioneers/threshold_avg.csv')

        def threshold_func(opt_price, inventory_level, time_until_replenish, threshold_avg, threshold_10percentile):
            # Ensure inventory_level is within the valid range (0 to 11)
            inventory_level = min(inventory_level, threshold_10percentile.shape[0] - 1)  # Should be between 0 and 11

            # Calculate the column index for time (ensure it's within bounds)
            column_index = min(20 - time_until_replenish, threshold_10percentile.shape[1] - 1)  # Make sure time index is valid
            
            # Now safely access the thresholds
            if opt_price < threshold_10percentile.iloc[inventory_level, column_index]:
                return threshold_avg.iloc[inventory_level, column_index]
            else:
                return opt_price

        # Predict optimal price
        optimal_price, demand_prediction = predict_optimal_price(
            pd.DataFrame([new_buyer_covariates], columns=['Covariate1', 'Covariate2', 'Covariate3']),
            prices_to_predict, rf_model
        )
        
        # Get the optimal price considering expected revenue over time and inventory
        opt_price = get_prices_over_time_and_expected_revenue_k(
            prices_to_predict, demand_prediction, T=21 - time_until_replenish, K=inventory_level
        )[0][0][-1]

        # Apply threshold function to adjust the price based on inventory and time
        opt_price = threshold_func(opt_price, inventory_level, time_until_replenish, threshold_avg, threshold_10percentile)
        
        return opt_price
        

