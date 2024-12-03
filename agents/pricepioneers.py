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
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        # Getting the inventory level of our agent (assumes inventories is a vector, so we take the first value)
        inventory_level = inventories[0]

        # Load training data to determine price range
        train_pricing_decisions = pd.read_csv('agents/pricepioneers/train_prices_decisions_2024.csv')
        min_price_threshold = np.percentile(train_pricing_decisions['price_item'], 10)
        prices_to_predict = np.arange(min_price_threshold, train_pricing_decisions['price_item'].max() + train_pricing_decisions['price_item'].mean(), 4)

        # Load Random Forest model for demand prediction
        with open('agents/pricepioneers/randomforrest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)

        # Predict demand for each price
        demand_predictions = []
        for price in prices_to_predict:
            # Prepare input data and predict demand probability using the Random Forest model
            input_data = pd.DataFrame([new_buyer_covariates], columns=['Covariate1', 'Covariate2', 'Covariate3'])
            price_data = pd.DataFrame({'price_item': [price]})
            expanded_data = pd.concat([input_data, price_data], axis=1)
            demand_prob = rf_model.predict_proba(expanded_data)[:, 1]  # Predicted probability of purchase
            demand_predictions.append(demand_prob)
        
        demand_predictions = np.array(demand_predictions)

        # Set up the linear programming problem for revenue maximization
        revenue = prices_to_predict * demand_predictions  # Revenue from each price option
        revenue = -revenue  # Negative because linprog minimizes the objective

        # Constraints: ensure no more than 12 products are sold for every 20 customers
        A_ub = np.ones((1, len(prices_to_predict)))  # All prices contribute to the sales
        b_ub = [12]  # 12 units available to sell

        # Solve the linear program
        result = linprog(revenue, A_ub=A_ub, b_ub=b_ub, method='highs')

        # Extract the optimal price from the result
        optimal_price = result.x[np.argmax(result.x)]  # Select the price that maximizes revenue

        #return optimal_price

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

        # Apply threshold function to adjust the price based on inventory and time
        opt_price = threshold_func(optimal_price, inventory_level, time_until_replenish, threshold_avg, threshold_10percentile)
        
        return opt_price
        

